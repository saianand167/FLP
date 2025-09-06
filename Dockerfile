# Start with a stable Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install all the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell Hugging Face which port the application will be running on
EXPOSE 7860

# The command to start the Gunicorn web server when the container starts
CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]