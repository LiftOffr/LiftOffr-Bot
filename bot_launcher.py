#!/usr/bin/env python3
"""
Bot Launcher

This script launches a minimal trading bot that avoids port conflicts
by explicitly setting a different port (5001) for any Flask components.
"""
import os
import sys
import subprocess

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" TRADING BOT LAUNCHER")
    print("=" * 60)
    print("\nThis launcher prevents port conflicts by using port 5001")
    
    # Set environment variables to prevent Flask conflicts
    env = dict(os.environ)
    env["TRADING_BOT_PROCESS"] = "1"
    env["FLASK_RUN_PORT"] = "5001"  # Force Flask to use port 5001 if it starts
    
    # Command to run the bot
    cmd = [sys.executable, "tiny_bot.py"]
    
    print("\nLaunching trading bot with isolated environment...")
    
    try:
        # Run the bot in a subprocess with the modified environment
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output from the bot
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        
        # Wait for the process to complete
        process.wait()
        print("\nTrading bot exited with code:", process.returncode)
        
    except KeyboardInterrupt:
        print("\nLauncher interrupted, stopping bot...")
        if 'process' in locals():
            process.terminate()
        
    except Exception as e:
        print(f"\nError launching trading bot: {e}")
    
    print("\n" + "=" * 60)
    print(" LAUNCHER EXITED")
    print("=" * 60)

if __name__ == "__main__":
    main()