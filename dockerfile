FROM python:3.13-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency file first to leverage Docker caching
COPY requirements.txt .

# 4. Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the local source code into the container
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py"]