#!/usr/bin/env python3

import os
import signal
import sys
import time
import traceback
from datetime import datetime

def test_signal_handling():
    """Simple test for SIGBUS signal handling"""
    
    # Setup log file
    os.makedirs('/tmp/log', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'/tmp/log/signal_test_{timestamp}.log'
    
    try:
        log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        header = f"Signal test started at {datetime.now()}\n"
        header += f"PID: {os.getpid()}\n"
        header += f"Signal mask: {signal.pthread_sigmask(signal.SIG_BLOCK, set())}\n"
        header += "=" * 50 + "\n"
        os.write(log_fd, header.encode('utf-8'))
        os.fsync(log_fd)
        print(f"Log file created: {log_file}")
    except Exception as e:
        print(f"Failed to create log file: {e}")
        return
    
    def handle_sigbus(signum, frame):
        """Handle SIGBUS signal"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            msg = f"\n[{timestamp}] *** SIGBUS RECEIVED ***\n"
            os.write(log_fd, msg.encode('utf-8'))
            
            # Capture stack trace
            try:
                tb_lines = traceback.format_stack(frame)
                os.write(log_fd, b"Stack trace:\n")
                for line in tb_lines[-10:]:
                    os.write(log_fd, line.encode('utf-8'))
            except:
                os.write(log_fd, b"Failed to capture traceback\n")
            
            os.fsync(log_fd)
            print(f"*** SIGBUS logged to {log_file} ***")
            
        except Exception as e:
            print(f"SIGBUS handler error: {e}")
        
        os._exit(128 + signal.SIGBUS)
    
    def handle_sigusr1(signum, frame):
        """Test signal handler"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            msg = f"[{timestamp}] SIGUSR1 received - handler working!\n"
            os.write(log_fd, msg.encode('utf-8'))
            os.fsync(log_fd)
            print("SIGUSR1 received - signal handler works!")
        except Exception as e:
            print(f"SIGUSR1 handler error: {e}")
    
    # Install signal handlers
    signal.signal(signal.SIGBUS, handle_sigbus)
    
    print("Signal handlers installed")
    
    # Test SIGUSR1 first
    print("Testing SIGUSR1...")
    os.kill(os.getpid(), signal.SIGUSR1)
    time.sleep(0.5)
    
    # Test SIGBUS directly
    print("Testing SIGBUS directly (this will terminate the process)...")
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGBUS)  # Send SIGBUS directly
    
    # This line should never be reached
    print("ERROR: Should not reach here!")

if __name__ == '__main__':
    test_signal_handling() 
