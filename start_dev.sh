#!/bin/bash

# Development startup script for Agents Flow
# Starts both the FastAPI backend and the React frontend concurrently
# Also runs embedding and tag generation processes

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to cleanup processes on exit
cleanup() {
    print_status "Stopping all development servers..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        print_status "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
        print_success "Backend stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        print_status "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
        print_success "Frontend stopped"
    fi
    
    print_success "All servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the project root directory."
    exit 1
fi

if [ ! -d "frontend" ]; then
    print_error "frontend directory not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Starting development servers..."
echo "=================================================="

# Check if virtual environment exists
if [ -d "sekai-env" ]; then
    PYTHON_CMD="sekai-env/bin/python"
    print_status "Using virtual environment: sekai-env"
else
    PYTHON_CMD="python3"
    print_warning "No virtual environment found, using system Python"
fi

# Run embedding and tag generation processes first
print_status "Running embedding and tag generation processes..."
echo "=================================================="

# Run batch tag generation
print_status "Generating tags for content..."
$PYTHON_CMD batch_generate_tags.py > tag_generation.log 2>&1
if [ $? -eq 0 ]; then
    print_success "Tag generation completed"
else
    print_warning "Tag generation had issues. Check tag_generation.log"
fi

# Run advanced embeddings
print_status "Running advanced embeddings..."
PYTHONPATH="$(pwd):$PYTHONPATH" $PYTHON_CMD -m embeddings.embed_contents_advanced > advanced_embeddings.log 2>&1
if [ $? -eq 0 ]; then
    print_success "Advanced embeddings completed"
else
    print_warning "Advanced embeddings had issues. Check advanced_embeddings.log"
fi

# Run combined embeddings
print_status "Running combined embeddings..."
PYTHONPATH="$(pwd):$PYTHONPATH" $PYTHON_CMD -m embeddings.embed_contents_combined > combined_embeddings.log 2>&1
if [ $? -eq 0 ]; then
    print_success "Combined embeddings completed"
else
    print_warning "Combined embeddings had issues. Check combined_embeddings.log"
fi

echo "=================================================="
print_status "Starting servers..."

# Start backend server
print_status "Starting FastAPI backend server..."
$PYTHON_CMD main.py > backend.log 2>&1 &
BACKEND_PID=$!
print_success "Backend started with PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    print_error "Backend failed to start. Check backend.log for details."
    exit 1
fi

# Start frontend server
print_status "Starting React frontend development server..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
print_success "Frontend started with PID: $FRONTEND_PID"

# Wait a moment for frontend to start
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend failed to start. Check frontend.log for details."
    cleanup
    exit 1
fi

echo ""
echo "=================================================="
print_success "Development servers are running!"
echo "📱 Frontend: http://localhost:5173 (or check frontend.log for port)"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "=================================================="
echo "Press Ctrl+C to stop all servers"
echo "=================================================="

# Function to show logs
show_logs() {
    echo ""
    echo "Recent backend logs:"
    echo "==================="
    tail -n 10 backend.log 2>/dev/null || echo "No backend logs yet"
    echo ""
    echo "Recent frontend logs:"
    echo "===================="
    tail -n 10 frontend.log 2>/dev/null || echo "No frontend logs yet"
    echo ""
}

# Show initial logs
show_logs

# Keep the script running and show logs periodically
while true; do
    sleep 10
    show_logs
done 