from langchain.text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from datetime import datetime
import logging
import numpy as np

class NestedChunkGenerator:
    """
    Creates nested chunks where each level is derived from the previous level's chunks.
    Each subsequent level contains smaller, more granular chunks.
    """
    
    def __init__(
        self,
        chunk_sizes: List[int] = None,
        overlap_ratios: List[float] = None,
        separators: List[str] = None
    ):
        """
        Initialize the nested chunk generator.
        
        Args:
            chunk_sizes: List of chunk sizes for each level (descending order)
            overlap_ratios: List of overlap ratios for each level
            separators: List of separators for text splitting
        """
        self.chunk_sizes = chunk_sizes or [2000, 1000, 500, 250]
        self.overlap_ratios = overlap_ratios or [0.2, 0.15, 0.1, 0.05]
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", " ", ""]
        
        # Validate inputs
        if not (len(self.chunk_sizes) == len(self.overlap_ratios)):
            raise ValueError("chunk_sizes and overlap_ratios must have the same length")
        
        # Set up logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the chunk generator."""
        logger = logging.getLogger("NestedChunkGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _create_level_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap_ratio: float
    ) -> List[str]:
        """
        Create chunks for a single level.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap_ratio: Overlap ratio between chunks
            
        Returns:
            List of text chunks
        """
        overlap_size = int(chunk_size * overlap_ratio)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=self.separators,
            length_function=len
        )
        
        return splitter.split_text(text)
    
    def generate_nested_chunks(self, text: str) -> Dict[str, Any]:
        """
        Generate nested chunks from the input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing nested chunks and metadata
        """
        self.logger.info(f"Starting nested chunk generation at {datetime.utcnow()}")
        
        nested_chunks = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "s3nh",  # Current user's login
                "total_levels": len(self.chunk_sizes),
                "original_text_length": len(text),
                "levels_info": {}
            },
            "chunks": {}
        }
        
        current_chunks = [text]
        
        for level, (chunk_size, overlap_ratio) in enumerate(
            zip(self.chunk_sizes, self.overlap_ratios), 1
        ):
            level_chunks = []
            level_metadata = {
                "chunk_size": chunk_size,
                "overlap_ratio": overlap_ratio,
                "num_chunks": 0,
                "avg_chunk_length": 0
            }
            
            # Process each chunk from the previous level
            for parent_chunk in current_chunks:
                child_chunks = self._create_level_chunks(
                    parent_chunk,
                    chunk_size,
                    overlap_ratio
                )
                level_chunks.extend(child_chunks)
            
            # Update metadata
            chunk_lengths = [len(chunk) for chunk in level_chunks]
            level_metadata.update({
                "num_chunks": len(level_chunks),
                "avg_chunk_length": np.mean(chunk_lengths),
                "min_chunk_length": min(chunk_lengths),
                "max_chunk_length": max(chunk_lengths)
            })
            
            # Store chunks and metadata
            nested_chunks["chunks"][f"level_{level}"] = level_chunks
            nested_chunks["metadata"]["levels_info"][f"level_{level}"] = level_metadata
            
            # Update current_chunks for next iteration
            current_chunks = level_chunks
            
            self.logger.info(
                f"Level {level} processing complete: {level_metadata['num_chunks']} chunks created"
            )
            
        return nested_chunks
    
    def get_chunk_hierarchy(
        self,
        nested_chunks: Dict[str, Any],
        chunk_index: int,
        level: int
    ) -> Dict[str, List[int]]:
        """
        Get the hierarchy of chunks (parent-child relationships) for a specific chunk.
        
        Args:
            nested_chunks: The nested chunks dictionary
            chunk_index: Index of the chunk to analyze
            level: Level of the chunk
            
        Returns:
            Dictionary containing parent and child chunk indices
        """
        hierarchy = {
            "parent_chunks": [],
            "child_chunks": []
        }
        
        # Find parent chunks (in previous levels)
        if level > 1:
            current_text = nested_chunks["chunks"][f"level_{level}"][chunk_index]
            for parent_level in range(level - 1, 0, -1):
                parent_chunks = nested_chunks["chunks"][f"level_{parent_level}"]
                for i, parent_chunk in enumerate(parent_chunks):
                    if current_text in parent_chunk:
                        hierarchy["parent_chunks"].append((parent_level, i))
                        break
        
        # Find child chunks (in next levels)
        if level < len(self.chunk_sizes):
            current_text = nested_chunks["chunks"][f"level_{level}"][chunk_index]
            for child_level in range(level + 1, len(self.chunk_sizes) + 1):
                child_chunks = nested_chunks["chunks"][f"level_{child_level}"]
                child_indices = [
                    i for i, chunk in enumerate(child_chunks)
                    if chunk in current_text
                ]
                if child_indices:
                    hierarchy["child_chunks"].extend(
                        [(child_level, idx) for idx in child_indices]
                    )
        
        return hierarchy

# Example usage
def main():
    # Sample text
    sample_text = """
    [Your long text here...]
    """
    
    # Initialize the chunk generator with custom sizes
    chunk_generator = NestedChunkGenerator(
        chunk_sizes=[1500, 750, 375, 180],
        overlap_ratios=[0.2, 0.15, 0.1, 0.05]
    )
    
    # Generate nested chunks
    result = chunk_generator.generate_nested_chunks(sample_text)
    
    # Print summary of each level
    print("\nNested Chunks Summary:")
    for level, metadata in result["metadata"]["levels_info"].items():
        print(f"\n{level.upper()}:")
        print(f"Number of chunks: {metadata['num_chunks']}")
        print(f"Average chunk length: {metadata['avg_chunk_length']:.2f}")
        print(f"Chunk size: {metadata['chunk_size']}")
        print(f"Overlap ratio: {metadata['overlap_ratio']}")
        
    # Get hierarchy for a specific chunk
    hierarchy = chunk_generator.get_chunk_hierarchy(result, chunk_index=0, level=2)
    print("\nChunk Hierarchy for level 2, index 0:")
    print(f"Parent chunks: {hierarchy['parent_chunks']}")
    print(f"Child chunks: {hierarchy['child_chunks']}")

if __name__ == "__main__":
    main()
