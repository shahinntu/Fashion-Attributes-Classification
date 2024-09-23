# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found. Please install it first."
    exit 1
fi

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Dependencies installed successfully."