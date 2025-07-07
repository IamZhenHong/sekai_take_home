#!/usr/bin/env python3
"""
Development startup script for Agents Flow
Starts both the FastAPI backend and the React frontend concurrently.
Also runs embedding and tag generation processes.
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Optional
import threading

class DevServer:
    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.running = False
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting FastAPI backend server...")
        
        # Check if virtual environment exists
        venv_path = Path("sekai-env")
        if venv_path.exists():
            # Use the virtual environment
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                python_path = venv_path / "bin" / "python"
        else:
            # Use system Python
            python_path = sys.executable
            
        try:
            self.backend_process = subprocess.Popen(
                [str(python_path), "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"✅ Backend started with PID: {self.backend_process.pid}")
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
        return True
    
    def start_frontend(self):
        """Start the React frontend development server"""
        print("🚀 Starting React frontend development server...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("❌ Frontend directory not found!")
            return False
            
        try:
            # Change to frontend directory and start npm dev
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"✅ Frontend started with PID: {self.frontend_process.pid}")
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
        return True
    
    def log_output(self, process: subprocess.Popen, prefix: str):
        """Log output from a process with a prefix"""
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"{prefix} {line.rstrip()}")
    
    def run_embeddings_and_tags(self):
        """Run embedding and tag generation processes"""
        print("🔧 Running embedding and tag generation processes...")
        print("=" * 50)
        
        # Check if virtual environment exists
        venv_path = Path("sekai-env")
        if venv_path.exists():
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                python_path = venv_path / "bin" / "python"
        else:
            python_path = sys.executable
            
        try:
            # Check if tags already exist and ask user about override
            import pandas as pd
            try:
                existing_df = pd.read_csv("contents_with_tags.csv")
                existing_tags_count = existing_df['generated_tags'].notna().sum()
                total_items = len(existing_df)
                
                if existing_tags_count > 0:
                    print(f"📊 Found existing tags: {existing_tags_count}/{total_items} items already have tags")
                    print("🤔 Do you want to override existing tags? (y/n): ", end="")
                    
                    # Get user input
                    override_choice = input().strip().lower()
                    override_flag = "--override" if override_choice in ['y', 'yes'] else ""
                    
                    if override_choice in ['y', 'yes']:
                        print("🔄 Will override existing tags")
                    else:
                        print("⏭️  Will skip items that already have tags")
                else:
                    override_flag = ""
                    print("📝 No existing tags found, will generate tags for all items")
            except FileNotFoundError:
                override_flag = ""
                print("📝 No existing tags file found, will generate tags for all items")
            
            # Run batch tag generation with progress tracking
            print("🏷️  Generating tags for content...")
            print("📊 This may take several minutes. Progress will be shown below:")
            print("-" * 50)
            
            # Start tag generation process with real-time output
            cmd = [str(python_path), "batch_generate_tags.py"]
            if override_flag:
                cmd.append(override_flag)
            
            tag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track progress in real-time
            batch_count = 0
            total_items = 0
            processed_items = 0
            
            if tag_process.stdout:
                for line in iter(tag_process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        print(f"[TAGS] {line}")
                        
                        # Track progress based on output
                        if "Processing" in line and "content items" in line:
                            # Extract total number of items
                            try:
                                total_items = int(line.split()[1])
                                print(f"📈 Total items to process: {total_items}")
                            except:
                                pass
                        elif "Processing batch" in line:
                            batch_count += 1
                            print(f"🔄 Processing batch {batch_count}")
                        elif "Item" in line and "Generated tags:" in line:
                            processed_items += 1
                            if total_items > 0:
                                progress = (processed_items / total_items) * 100
                                print(f"📊 Progress: {processed_items}/{total_items} ({progress:.1f}%)")
            
            # Wait for process to complete
            tag_process.wait()
            
            if tag_process.returncode == 0:
                print("✅ Tag generation completed successfully!")
            else:
                print(f"⚠️  Tag generation had issues (return code: {tag_process.returncode})")
                
            # Run advanced embeddings
            print("\n🔍 Running advanced embeddings...")
            # Set PYTHONPATH to include current directory
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd()) + os.pathsep + env.get('PYTHONPATH', '')
            
            # Check if contents_with_tags.csv exists before running embeddings
            csv_path = Path("contents_with_tags.csv")
            if not csv_path.exists():
                csv_path = Path("datasets/contents_with_tags.csv")
                if not csv_path.exists():
                    print("⚠️  contents_with_tags.csv not found in current directory or datasets/. Skipping embeddings.")
                else:
                    print(f"✅ Found contents_with_tags.csv in {csv_path}")
            else:
                print(f"✅ Found contents_with_tags.csv in {csv_path}")
            
            if csv_path.exists():
                advanced_process = subprocess.run(
                    [str(python_path), "-m", "embeddings.embed_contents_advanced"],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                    cwd=str(Path.cwd()),  # Ensure we're in the project root
                    env=env
                )
                if advanced_process.returncode == 0:
                    print("✅ Advanced embeddings completed")
                else:
                    print(f"⚠️  Advanced embeddings had issues: {advanced_process.stderr}")
                    
                # Run combined embeddings
                print("🔗 Running combined embeddings...")
                combined_process = subprocess.run(
                    [str(python_path), "-m", "embeddings.embed_contents_combined"],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                    cwd=str(Path.cwd()),  # Ensure we're in the project root
                    env=env
                )
                if combined_process.returncode == 0:
                    print("✅ Combined embeddings completed")
                else:
                    print(f"⚠️  Combined embeddings had issues: {combined_process.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Embedding/tag generation processes timed out")
        except Exception as e:
            print(f"❌ Error running embedding/tag processes: {e}")
            
        print("=" * 50)
        return True

    def start_servers(self):
        """Start both servers concurrently"""
        print("🎯 Starting development servers...")
        print("=" * 50)
        
        # Run embeddings and tags first
        self.run_embeddings_and_tags()
        
        # Start backend
        if not self.start_backend():
            return False
            
        # Start frontend
        if not self.start_frontend():
            self.stop_backend()
            return False
        
        self.running = True
        
        # Start output logging in separate threads
        if self.backend_process:
            backend_thread = threading.Thread(
                target=self.log_output,
                args=(self.backend_process, "[BACKEND]"),
                daemon=True
            )
            backend_thread.start()
            
        if self.frontend_process:
            frontend_thread = threading.Thread(
                target=self.log_output,
                args=(self.frontend_process, "[FRONTEND]"),
                daemon=True
            )
            frontend_thread.start()
        
        print("\n" + "=" * 50)
        print("🎉 Development servers are running!")
        print("📱 Frontend: http://localhost:5173 (or check npm output for port)")
        print("🔧 Backend:  http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop all servers")
        print("=" * 50)
        
        return True
    
    def stop_backend(self):
        """Stop the backend server"""
        if self.backend_process:
            print("🛑 Stopping backend server...")
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Backend didn't stop gracefully, forcing...")
                self.backend_process.kill()
            except Exception as e:
                print(f"❌ Error stopping backend: {e}")
            finally:
                self.backend_process = None
    
    def stop_frontend(self):
        """Stop the frontend server"""
        if self.frontend_process:
            print("🛑 Stopping frontend server...")
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Frontend didn't stop gracefully, forcing...")
                self.frontend_process.kill()
            except Exception as e:
                print(f"❌ Error stopping frontend: {e}")
            finally:
                self.frontend_process = None
    
    def stop_all(self):
        """Stop all servers"""
        print("\n🛑 Stopping all development servers...")
        self.running = False
        self.stop_frontend()
        self.stop_backend()
        print("✅ All servers stopped")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\n📡 Received signal {signum}")
        self.stop_all()
        sys.exit(0)

def main():
    """Main function to start the development servers"""
    server = DevServer()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, server.signal_handler)
    signal.signal(signal.SIGTERM, server.signal_handler)
    
    try:
        if server.start_servers():
            # Keep the main thread alive
            while server.running:
                time.sleep(1)
        else:
            print("❌ Failed to start servers")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n📡 Keyboard interrupt received")
        server.stop_all()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        server.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main() 