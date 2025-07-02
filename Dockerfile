 # Dockerfile
FROM python:3.9-slim

# 메타데이터
LABEL maintainer="Satellite Game Theory Research Team"
LABEL description="Satellite Pursuit-Evasion Game Environment"
LABEL version="1.0.0"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 복사
COPY . .

# 패키지 설치
RUN pip install -e .

# 포트 노출 (Jupyter 등을 위해)
EXPOSE 8888

# 기본 명령어
CMD ["python", "main.py", "--help"]

