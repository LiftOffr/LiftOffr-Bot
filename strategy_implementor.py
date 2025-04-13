#!/usr/bin/env python3
"""
Strategy Implementor for Kraken Trading Bot

This module implements the recommendations from strategy_optimizer.py
by generating updated strategy files with optimized parameters.
"""

import os
import sys
import json
import logging
import re
from datetime import datetime
import traceback
import inspect
import importlib
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
BACKUP_DIR = "strategy_backups"
OPTIMIZATION_RESULTS_DIR = "optimization_results"
IMPLEMENTATION_DIR = "strategy_implementations"
DEFAULT_RECOMMENDATIONS_FILE = "latest_strategy_improvements.json"

# Helper functions
def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def create_timestamp():
    """Create a timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class StrategyImplementor:
    """
    Implement strategy improvements recommended by the optimizer
    by modifying existing strategy files or creating new ones.
    """
    
    def __init__(self, recommendations_file=None):
        """
        Initialize the strategy implementor
        
        Args:
            recommendations_file (str): Path to recommendations JSON file
        """
        self.recommendations = None
        self.strategies_to_improve = {}
        self.strategies_to_remove = []
        self.backup_dir = BACKUP_DIR
        self.implementation_dir = IMPLEMENTATION_DIR
        self.timestamp = create_timestamp()
        
        # Ensure directories exist
        ensure_directory(self.backup_dir)
        ensure_directory(self.implementation_dir)
        
        # Load recommendations
        if recommendations_file:
            self.load_recommendations(recommendations_file)
        else:
            self._find_latest_recommendations()
    
    def _find_latest_recommendations(self):
        """Find the latest recommendations file in optimization_results directory"""
        if not os.path.exists(OPTIMIZATION_RESULTS_DIR):
            logger.error(f"Optimization results directory not found: {OPTIMIZATION_RESULTS_DIR}")
            return False
            
        # List files with strategy_improvements in the name
        files = [f for f in os.listdir(OPTIMIZATION_RESULTS_DIR) 
                if f.startswith("strategy_improvements_") and f.endswith(".json")]
        
        if not files:
            logger.error("No strategy improvement files found.")
            return False
            
        # Sort by timestamp in filename
        files.sort(reverse=True)
        latest_file = os.path.join(OPTIMIZATION_RESULTS_DIR, files[0])
        
        logger.info(f"Using latest recommendations file: {latest_file}")
        return self.load_recommendations(latest_file)
    
    def load_recommendations(self, file_path):
        """
        Load strategy improvement recommendations
        
        Args:
            file_path (str): Path to recommendations JSON file
            
        Returns:
            bool: Success status
        """
        try:
            with open(file_path, 'r') as f:
                recommendations = json.load(f)
                
            self.recommendations = recommendations
            self.strategies_to_improve = recommendations.get('to_improve', {})
            self.strategies_to_remove = recommendations.get('to_remove', [])
            
            # Save a copy as latest
            latest_path = os.path.join(OPTIMIZATION_RESULTS_DIR, DEFAULT_RECOMMENDATIONS_FILE)
            shutil.copy2(file_path, latest_path)
            
            logger.info(f"Loaded recommendations: {len(self.strategies_to_improve)} to improve, " 
                       f"{len(self.strategies_to_remove)} to remove")
            return True
            
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
            traceback.print_exc()
            return False
    
    def locate_strategy_file(self, strategy_name):
        """
        Locate the file containing a strategy class
        
        Args:
            strategy_name (str): Name of strategy class
            
        Returns:
            str: Path to strategy file, or None if not found
        """
        # Check common strategy files
        common_strategy_files = [
            f"{strategy_name.lower()}.py",
            f"{strategy_name.lower()}_strategy.py",
            "trading_strategy.py",
            "strategies.py"
        ]
        
        for filename in common_strategy_files:
            if os.path.exists(filename):
                # Check if the file contains the strategy class
                with open(filename, 'r') as f:
                    content = f.read()
                    if f"class {strategy_name}" in content:
                        logger.info(f"Found strategy {strategy_name} in {filename}")
                        return filename
        
        # If not found, search all Python files in current directory
        for filename in os.listdir('.'):
            if filename.endswith('.py') and filename not in common_strategy_files:
                with open(filename, 'r') as f:
                    content = f.read()
                    if f"class {strategy_name}" in content:
                        logger.info(f"Found strategy {strategy_name} in {filename}")
                        return filename
        
        logger.warning(f"Could not find file containing strategy {strategy_name}")
        return None
    
    def backup_strategy_file(self, file_path):
        """
        Create a backup of a strategy file
        
        Args:
            file_path (str): Path to file
            
        Returns:
            str: Path to backup file
        """
        filename = os.path.basename(file_path)
        backup_path = os.path.join(self.backup_dir, f"{filename}.{self.timestamp}")
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def extract_strategy_class(self, file_path, strategy_name):
        """
        Extract the strategy class definition from a file
        
        Args:
            file_path (str): Path to file
            strategy_name (str): Name of strategy class
            
        Returns:
            tuple: (class_content, start_line, end_line)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        class_start = None
        class_end = None
        in_class = False
        indent_level = 0
        
        # Find class definition
        for i, line in enumerate(lines):
            if f"class {strategy_name}" in line:
                class_start = i
                in_class = True
                # Determine indent level for class definition (usually 0)
                indent_level = len(line) - len(line.lstrip())
                continue
                
            if in_class:
                # Check for next class or end of file
                stripped = line.lstrip()
                if stripped and len(line) - len(stripped) <= indent_level:
                    if line.startswith('class ') or line.startswith('def '):
                        class_end = i
                        break
        
        # If we didn't find the end, it's the end of the file
        if class_end is None:
            class_end = len(lines)
        
        if class_start is not None:
            class_content = ''.join(lines[class_start:class_end])
            return class_content, class_start, class_end
        else:
            logger.error(f"Could not find class {strategy_name} in {file_path}")
            return None, None, None
    
    def update_parameter_value(self, class_content, param_name, new_value):
        """
        Update a parameter value in the strategy class constructor
        
        Args:
            class_content (str): Class definition
            param_name (str): Parameter name
            new_value: New parameter value
            
        Returns:
            str: Updated class content
        """
        # Find the constructor method
        init_pattern = r"def\s+__init__\s*\("
        init_match = re.search(init_pattern, class_content)
        
        if not init_match:
            logger.warning(f"Could not find constructor method for parameter {param_name}")
            return class_content
        
        # Find the parameter assignment pattern
        param_pattern = rf"self\.{param_name}\s*=\s*([^,)\n]+)"
        param_match = re.search(param_pattern, class_content)
        
        if param_match:
            # Get the current value and format the new value to match it
            current_value = param_match.group(1).strip()
            
            # Format new value based on type of current value
            formatted_value = self._format_value(current_value, new_value)
            
            # Replace the parameter value
            updated_content = re.sub(
                param_pattern, 
                f"self.{param_name} = {formatted_value}", 
                class_content
            )
            
            logger.info(f"Updated parameter {param_name}: {current_value} -> {formatted_value}")
            return updated_content
        else:
            # Try to find the parameter in the constructor arguments
            init_args_pattern = rf"def\s+__init__.*?{param_name}\s*=\s*([^,)]+)"
            init_args_match = re.search(init_args_pattern, class_content)
            
            if init_args_match:
                current_value = init_args_match.group(1).strip()
                formatted_value = self._format_value(current_value, new_value)
                
                updated_content = re.sub(
                    init_args_pattern,
                    f"def __init__.*?{param_name}={formatted_value}",
                    class_content
                )
                
                logger.info(f"Updated constructor parameter {param_name}: {current_value} -> {formatted_value}")
                return updated_content
            else:
                logger.warning(f"Could not find parameter {param_name} to update")
                return class_content
    
    def _format_value(self, current_value, new_value):
        """
        Format a new value to match the type of the current value
        
        Args:
            current_value (str): String representation of current value
            new_value: New value (could be any type)
            
        Returns:
            str: Formatted value as string
        """
        # Check if it's a tuple
        if current_value.startswith('(') and current_value.endswith(')'):
            if isinstance(new_value, tuple):
                return str(new_value)
            elif isinstance(new_value, str) and new_value.startswith('(') and new_value.endswith(')'):
                return new_value
            else:
                return f"({new_value})"
        
        # Check if it's a float
        if '.' in current_value:
            try:
                float_val = float(current_value)
                if isinstance(new_value, (int, float)):
                    return str(float(new_value))
                else:
                    return str(float(eval(str(new_value))))
            except:
                pass
        
        # Check if it's an integer
        try:
            int_val = int(current_value)
            if isinstance(new_value, int):
                return str(new_value)
            else:
                return str(int(eval(str(new_value))))
        except:
            pass
        
        # Default to string representation
        if isinstance(new_value, (list, tuple, dict)):
            return str(new_value)
        else:
            return str(new_value)
    
    def implement_parameter_improvements(self, strategy_name, recommendations):
        """
        Implement parameter improvements for a strategy
        
        Args:
            strategy_name (str): Strategy name
            recommendations (dict): Parameter recommendations
            
        Returns:
            tuple: (success, file_path)
        """
        # Find the strategy file
        file_path = self.locate_strategy_file(strategy_name)
        if not file_path:
            return False, None
        
        # Create a backup
        backup_path = self.backup_strategy_file(file_path)
        if not backup_path:
            return False, None
        
        # Extract the strategy class
        class_content, start_line, end_line = self.extract_strategy_class(file_path, strategy_name)
        if not class_content:
            return False, None
        
        updated_content = class_content
        
        # Update parameters
        for param_name, param_info in recommendations.items():
            # Get the new value
            if "value" in param_info:
                new_value = param_info["value"]
            elif "range" in param_info:
                # Use the middle value of the range
                range_values = param_info["range"]
                if isinstance(range_values, list) and len(range_values) >= 2:
                    if isinstance(range_values[0], (int, float)) and isinstance(range_values[1], (int, float)):
                        new_value = (range_values[0] + range_values[1]) / 2
                    else:
                        new_value = range_values[0]  # Default to first value if not numeric
                else:
                    logger.warning(f"Invalid range format for {param_name}: {range_values}")
                    continue
            else:
                logger.warning(f"No value or range provided for {param_name}")
                continue
            
            # Update the parameter in the class content
            updated_content = self.update_parameter_value(updated_content, param_name, new_value)
        
        # Create a new implementation file
        impl_file = os.path.join(
            self.implementation_dir, 
            f"{strategy_name.lower()}_optimized.py"
        )
        
        # Read the original file
        with open(file_path, 'r') as f:
            original_content = f.readlines()
        
        # Replace the class definition
        new_content = (
            original_content[:start_line] + 
            [updated_content] + 
            original_content[end_line:]
        )
        
        # Write the new implementation
        with open(impl_file, 'w') as f:
            f.writelines(new_content)
        
        logger.info(f"Created optimized strategy implementation: {impl_file}")
        
        # Create a strategy parameter report
        param_report = self._create_parameter_report(strategy_name, recommendations, class_content, updated_content)
        report_file = os.path.join(
            self.implementation_dir, 
            f"{strategy_name.lower()}_parameter_changes.txt"
        )
        
        with open(report_file, 'w') as f:
            f.write(param_report)
        
        logger.info(f"Created parameter changes report: {report_file}")
        
        return True, impl_file
    
    def _create_parameter_report(self, strategy_name, recommendations, old_content, new_content):
        """
        Create a report of parameter changes
        
        Args:
            strategy_name (str): Strategy name
            recommendations (dict): Parameter recommendations
            old_content (str): Original class content
            new_content (str): Updated class content
            
        Returns:
            str: Parameter change report
        """
        report = [
            f"Parameter Changes for {strategy_name}",
            "=" * 50,
            ""
        ]
        
        for param_name, param_info in recommendations.items():
            # Extract current value
            param_pattern = rf"self\.{param_name}\s*=\s*([^,)\n]+)"
            old_match = re.search(param_pattern, old_content)
            new_match = re.search(param_pattern, new_content)
            
            if old_match and new_match:
                old_value = old_match.group(1).strip()
                new_value = new_match.group(1).strip()
                
                change_type = param_info.get("change", "update")
                
                # Add to report
                report.append(f"Parameter: {param_name}")
                report.append(f"Old value: {old_value}")
                report.append(f"New value: {new_value}")
                report.append(f"Change type: {change_type}")
                
                if "range" in param_info:
                    report.append(f"Recommended range: {param_info['range']}")
                
                report.append("")
        
        return "\n".join(report)
    
    def implement_all_improvements(self):
        """
        Implement all strategy improvements
        
        Returns:
            dict: Results of implementation
        """
        results = {
            "improved": [],
            "removed": [],
            "failed": []
        }
        
        # Implement improvements
        for strategy_name, recommendations in self.strategies_to_improve.items():
            success, file_path = self.implement_parameter_improvements(strategy_name, recommendations)
            
            if success:
                results["improved"].append({
                    "strategy": strategy_name,
                    "file": file_path
                })
            else:
                results["failed"].append({
                    "strategy": strategy_name,
                    "reason": "Implementation failed"
                })
        
        # Process removals
        for strategy_name in self.strategies_to_remove:
            file_path = self.locate_strategy_file(strategy_name)
            
            if file_path:
                # Create a backup
                backup_path = self.backup_strategy_file(file_path)
                
                if backup_path:
                    # Create a "removed" version that disables the strategy
                    self._create_disabled_strategy(strategy_name, file_path)
                    
                    results["removed"].append({
                        "strategy": strategy_name,
                        "file": file_path,
                        "backup": backup_path
                    })
                else:
                    results["failed"].append({
                        "strategy": strategy_name,
                        "reason": "Backup failed"
                    })
            else:
                results["failed"].append({
                    "strategy": strategy_name,
                    "reason": "Strategy file not found"
                })
        
        return results
    
    def _create_disabled_strategy(self, strategy_name, file_path):
        """
        Create a disabled version of a strategy
        
        Args:
            strategy_name (str): Strategy name
            file_path (str): Path to strategy file
            
        Returns:
            bool: Success status
        """
        # Extract the strategy class
        class_content, start_line, end_line = self.extract_strategy_class(file_path, strategy_name)
        if not class_content:
            return False
        
        # Modify the class to disable trading
        disabled_content = class_content
        
        # Override should_buy and should_sell methods
        sell_pattern = r"def\s+should_sell\s*\(.*?\).*?:"
        buy_pattern = r"def\s+should_buy\s*\(.*?\).*?:"
        
        # Replace should_buy with always returning False
        if re.search(buy_pattern, disabled_content):
            disabled_content = re.sub(
                buy_pattern + r".*?(def|\Z)", 
                f"def should_buy(self) -> bool:\n        # DISABLED: Strategy removed due to poor performance\n        return False\n\n    def",
                disabled_content, 
                flags=re.DOTALL
            )
        
        # Replace should_sell with always returning False
        if re.search(sell_pattern, disabled_content):
            disabled_content = re.sub(
                sell_pattern + r".*?(def|\Z)", 
                f"def should_sell(self) -> bool:\n        # DISABLED: Strategy removed due to poor performance\n        return False\n\n    def",
                disabled_content, 
                flags=re.DOTALL
            )
        
        # Add a comment to the class definition
        class_def_pattern = rf"class\s+{strategy_name}\s*\(.*?\):"
        disabled_content = re.sub(
            class_def_pattern,
            f"class {strategy_name}(\\1):  # DISABLED: Strategy removed due to poor performance",
            disabled_content
        )
        
        # Create an implementation file for the disabled strategy
        impl_file = os.path.join(
            self.implementation_dir, 
            f"{strategy_name.lower()}_disabled.py"
        )
        
        # Read the original file
        with open(file_path, 'r') as f:
            original_content = f.readlines()
        
        # Replace the class definition
        new_content = (
            original_content[:start_line] + 
            [disabled_content] + 
            original_content[end_line:]
        )
        
        # Write the disabled implementation
        with open(impl_file, 'w') as f:
            f.writelines(new_content)
        
        logger.info(f"Created disabled strategy implementation: {impl_file}")
        return True
    
    def create_deployment_guide(self, results):
        """
        Create a deployment guide for the strategy improvements
        
        Args:
            results (dict): Implementation results
            
        Returns:
            str: Path to deployment guide
        """
        guide_file = os.path.join(
            self.implementation_dir, 
            f"deployment_guide_{self.timestamp}.md"
        )
        
        guide = [
            "# Strategy Optimization Deployment Guide",
            "",
            f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This guide provides instructions for deploying the strategy optimizations.",
            "",
            "## Summary",
            "",
            f"- Improved Strategies: {len(results['improved'])}",
            f"- Removed Strategies: {len(results['removed'])}",
            f"- Failed Actions: {len(results['failed'])}",
            "",
            "## Improved Strategies",
            ""
        ]
        
        for item in results["improved"]:
            guide.append(f"### {item['strategy']}")
            guide.append("")
            guide.append(f"New implementation: `{item['file']}`")
            guide.append("")
            guide.append("To deploy this improvement:")
            guide.append("")
            guide.append(f"1. Review the changes in `{os.path.basename(item['file'])}`")
            guide.append("2. Test the strategy with the new parameters")
            guide.append(f"3. If satisfied, replace the original strategy code with the optimized version")
            guide.append("")
        
        guide.append("## Removed Strategies")
        guide.append("")
        
        for item in results["removed"]:
            guide.append(f"### {item['strategy']}")
            guide.append("")
            guide.append(f"Original file: `{item['file']}`")
            guide.append(f"Backup created: `{item['backup']}`")
            guide.append("")
            guide.append("This strategy has been marked for removal due to poor performance.")
            guide.append("")
            guide.append("To deploy this change:")
            guide.append("")
            guide.append("1. Option 1: Disable the strategy in your trading bot configuration")
            guide.append("2. Option 2: Replace the original with the disabled version")
            guide.append("")
        
        if results["failed"]:
            guide.append("## Failed Actions")
            guide.append("")
            
            for item in results["failed"]:
                guide.append(f"- {item['strategy']}: {item['reason']}")
            
            guide.append("")
        
        guide.append("## Verification Steps")
        guide.append("")
        guide.append("After deploying changes, monitor the trading bot performance:")
        guide.append("")
        guide.append("1. Check that the bot starts without errors")
        guide.append("2. Verify that the modified strategies are generating appropriate signals")
        guide.append("3. Monitor the first few trades to ensure they are executing as expected")
        guide.append("4. Track performance metrics over time to confirm improvements")
        
        # Write the guide
        with open(guide_file, 'w') as f:
            f.write("\n".join(guide))
        
        logger.info(f"Created deployment guide: {guide_file}")
        return guide_file


def main():
    """
    Main function to run the strategy implementor
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Implement strategy improvements")
    parser.add_argument("--recommendations", "-r", type=str, help="Path to recommendations JSON file")
    args = parser.parse_args()
    
    # Create implementor
    implementor = StrategyImplementor(args.recommendations)
    
    # Implement improvements
    if implementor.recommendations:
        results = implementor.implement_all_improvements()
        
        # Create deployment guide
        guide_file = implementor.create_deployment_guide(results)
        
        # Print summary
        print("\nStrategy Implementation Summary:")
        print("===============================")
        print(f"Improved: {len(results['improved'])} strategies")
        print(f"Removed: {len(results['removed'])} strategies")
        print(f"Failed: {len(results['failed'])} actions")
        print(f"\nSee deployment guide: {guide_file}")
    else:
        print("No recommendations loaded. Exiting.")


if __name__ == "__main__":
    main()