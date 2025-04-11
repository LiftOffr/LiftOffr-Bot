#!/usr/bin/env python3

import os
import sys
import time
import subprocess

def main():
    while True:
        try:
            user_input = input("\nEnter command (CURRENT, STATUS, EXIT): ").strip().upper()
            
            if user_input == "CURRENT" or user_input == "STATUS":
                # Run the get_current_status.py script to display current status
                print("\nFetching latest trading status...\n")
                subprocess.run(["python", "get_current_status.py"])
            elif user_input == "EXIT":
                print("Exiting command interface...")
                break
            else:
                print("Unknown command. Available commands: CURRENT, STATUS, EXIT")
        except KeyboardInterrupt:
            print("\nExiting command interface...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KRAKEN TRADING BOT - COMMAND INTERFACE")
    print("=" * 80)
    print("Type 'CURRENT' or 'STATUS' to display current trading status")
    print("Type 'EXIT' to exit this interface")
    print("=" * 80)
    
    main()