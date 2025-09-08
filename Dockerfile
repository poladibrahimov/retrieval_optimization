# Use a slim version of Python for a smaller image size.
FROM python:3.13-slim

# Set the working directory.
WORKDIR /app

# Copy only requirements to leverage Docker cache.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Set the default command.
CMD ["python", "optimize.py"]
