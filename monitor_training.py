#!/usr/bin/env python3

import os
import time
import psutil
import signal
import sys
from datetime import datetime

def monitor_training_process(parent_pid, timeout_seconds=300):
    """Monitor a PyTorch distributed training process and its children"""
    
    print(f"Monitoring training process {parent_pid} and children...")
    print(f"Timeout: {timeout_seconds} seconds")
    print(f"Log directory: /tmp/log/")
    
    os.makedirs('/tmp/log', exist_ok=True)
    log_file = f'/tmp/log/training_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    try:
        parent = psutil.Process(parent_pid)
        print(f"Found parent process: {parent.name()} (PID: {parent_pid})")
        
        with open(log_file, 'w') as f:
            f.write(f"Training Monitor Started: {datetime.now()}\n")
            f.write(f"Parent PID: {parent_pid}\n")
            f.write(f"Timeout: {timeout_seconds} seconds\n")
            f.write("=" * 50 + "\n")
            
            start_time = time.time()
            last_check = start_time
            
            while True:
                current_time = time.time()
                
                # Check if parent still exists
                if not parent.is_running():
                    msg = f"[{datetime.now()}] Parent process {parent_pid} terminated\n"
                    f.write(msg)
                    f.flush()
                    print(msg.strip())
                    break
                
                # Get all child processes
                try:
                    children = parent.children(recursive=True)
                    child_info = []
                    
                    for child in children:
                        try:
                            status = child.status()
                            cpu_percent = child.cpu_percent()
                            memory_mb = child.memory_info().rss / 1024 / 1024
                            child_info.append({
                                'pid': child.pid,
                                'name': child.name(),
                                'status': status,
                                'cpu': cpu_percent,
                                'memory_mb': memory_mb
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            child_info.append({
                                'pid': child.pid,
                                'name': 'CRASHED',
                                'status': 'CRASHED',
                                'cpu': 0,
                                'memory_mb': 0
                            })
                
                    # Log every 30 seconds
                    if current_time - last_check >= 30:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        f.write(f"\n[{timestamp}] Status check - {len(children)} child processes:\n")
                        
                        for info in child_info:
                            f.write(f"  PID {info['pid']}: {info['name']} - {info['status']} "
                                   f"(CPU: {info['cpu']:.1f}%, Memory: {info['memory_mb']:.1f} MB)\n")
                        
                        f.flush()
                        print(f"[{timestamp}] Monitoring {len(children)} child processes...")
                        
                        # Check for crashed children
                        crashed = [info for info in child_info if info['status'] == 'CRASHED']
                        if crashed:
                            crash_msg = f"[{timestamp}] DETECTED CRASHED PROCESSES: {[c['pid'] for c in crashed]}\n"
                            f.write(crash_msg)
                            f.flush()
                            print(crash_msg.strip())
                        
                        last_check = current_time
                
                    # Check timeout
                    if current_time - start_time > timeout_seconds:
                        timeout_msg = f"[{datetime.now()}] TIMEOUT after {timeout_seconds} seconds - process may be hanging\n"
                        f.write(timeout_msg)
                        f.flush()
                        print(timeout_msg.strip())
                        break
                    
                except psutil.NoSuchProcess:
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
    except psutil.NoSuchProcess:
        print(f"Process {parent_pid} not found")
        return 1
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped by user")
        return 0
    
    print(f"Monitoring complete. Log saved to: {log_file}")
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <parent_pid> [timeout_seconds]")
        print("Example: python monitor_training.py 12345 600")
        sys.exit(1)
    
    parent_pid = int(sys.argv[1])
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    sys.exit(monitor_training_process(parent_pid, timeout)) 