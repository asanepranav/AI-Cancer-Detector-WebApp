# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures we don't store unnecessary files
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
# This is the standard port for Hugging Face Spaces
EXPOSE 7860

# Define the command to run your app using Gunicorn
# This will run your Flask app on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "flask_app:app"]
