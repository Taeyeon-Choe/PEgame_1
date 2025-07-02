"""
위성 추격-회피 게임 프로젝트 설치 스크립트
"""

from setuptools import setup, find_packages

# 버전 정보
VERSION = "1.0.0"

# README 파일 읽기
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt 파일 읽기
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satellite-pursuit-evasion",
    version=VERSION,
    author="Satellite Game Theory Research Team",
    author_email="contact@example.com",
    description="위성 추격-회피 게임을 위한 강화학습 프레임워크",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/satellite-pursuit-evasion",
    project_urls={
        "Bug Tracker": "https://github.com/username/satellite-pursuit-evasion/issues",
        "Documentation": "https://satellite-pursuit-evasion.readthedocs.io/",
        "Source Code": "https://github.com/username/satellite-pursuit-evasion",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "satellite-game=main:main",
            "quick-train=main:quick_train",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.json"],
        "": ["*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "reinforcement learning",
        "satellite",
        "orbital mechanics", 
        "game theory",
        "nash equilibrium",
        "pursuit evasion",
        "space",
        "deep learning",
        "SAC",
        "zero-sum game"
    ],
    
    # 메타데이터
    license="MIT",
    platforms=["any"],
    
    # 의존성 관리
    dependency_links=[],
    
    # 테스트 설정
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
    ],
    
    # 추가 설정
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python",
        },
    },
)