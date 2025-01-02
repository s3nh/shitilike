import time
import numpy as np
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich import box
from rich.console import Console
import math

class TrainingMonitor:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        
        # Initialize metrics history
        self.loss_history = []
        self.accuracy_history = []
        self.lr_history = []
        
        # Configure layout
        self.layout.split_column(
            Layout(name="upper"),
            Layout(name="lower")
        )
        self.layout["upper"].split_row(
            Layout(name="metrics"),
            Layout(name="plot")
        )
        
    def create_metrics_panel(self, epoch, loss, accuracy, learning_rate):
        """Create panel with current training metrics"""
        metrics_table = Table.grid(padding=1)
        metrics_table.add_row("Epoch:", f"[yellow]{epoch}")
        metrics_table.add_row("Loss:", f"[red]{loss:.4f}")
        metrics_table.add_row("Accuracy:", f"[green]{accuracy:.2%}")
        metrics_table.add_row("Learning Rate:", f"[blue]{learning_rate:.6f}")
        return Panel(metrics_table, title="Training Metrics", border_style="bright_blue")
    
    def create_progress_panel(self, current_step, total_steps):
        """Create progress bar panel"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        )
        task = progress.add_task("Training Progress", total=total_steps)
        progress.update(task, completed=current_step)
        return Panel(progress, border_style="bright_blue")
    
    def create_plot(self, values, width=50, height=10, title=""):
        """Create ASCII plot of training metrics"""
        if not values:
            return ""
            
        # Normalize values to fit in height
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        normalized = [(v - min_val) / range_val * (height - 1) for v in values]
        
        # Create plot
        plot = [[" " for _ in range(width)] for _ in range(height)]
        
        # Plot points
        x_step = (len(values) - 1) / (width - 1) if len(values) > 1 else 1
        for x in range(width):
            if x * x_step >= len(values):
                break
            y = normalized[int(x * x_step)]
            plot[height - 1 - int(y)][x] = "•"
            
        # Add axes
        for i in range(height):
            plot[i][0] = "│"
        plot[-1] = ["─" for _ in range(width)]
        plot[-1][0] = "└"
        
        # Convert to string
        return Panel(
            "\n".join("".join(row) for row in plot),
            title=f"[bright_blue]{title}[/]",
            border_style="bright_blue"
        )
    
    def update_display(self, epoch, loss, accuracy, lr, current_step, total_steps):
        """Update the live display with current training metrics"""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.lr_history.append(lr)
        
        # Update layout sections
        self.layout["upper"]["metrics"].update(
            self.create_metrics_panel(epoch, loss, accuracy, lr)
        )
        self.layout["upper"]["plot"].update(
            self.create_plot(self.loss_history[-50:], title="Loss History")
        )
        self.layout["lower"].update(
            self.create_progress_panel(current_step, total_steps)
        )
        
    def start_monitoring(self, total_epochs=10, steps_per_epoch=100):
        """Start the live monitoring display"""
        with Live(self.layout, refresh_per_second=4) as live:
            for epoch in range(total_epochs):
                for step in range(steps_per_epoch):
                    # Simulate training metrics (replace with actual training data)
                    loss = 2 * math.exp(-0.1 * (epoch * steps_per_epoch + step)) + 0.2 * np.random.random()
                    accuracy = 1 - loss/3 + 0.05 * np.random.random()
                    lr = 0.001 * math.exp(-0.1 * epoch)
                    
                    self.update_display(
                        epoch + 1,
                        loss,
                        accuracy,
                        lr,
                        epoch * steps_per_epoch + step,
                        total_epochs * steps_per_epoch
                    )
                    time.sleep(0.1)  # Simulate training time

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.start_monitoring()
