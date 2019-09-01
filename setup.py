import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visionmadeeasy",
    version="2019.02.17",
    author="Paul Baumgarten",
    author_email="pbaumgarten@gmail.com",
    description="A module intended to abstract away a lot of the complexity of using OpenCV to detect and recognise faces for beginner programmers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pbaumgarten.com/visionmadeeasy",
    packages=setuptools.find_packages(),
    keywords='opencv face detection recognition beginner',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['opencv-contrib-python','numpy','Pillow'],
    python_requires='>=3'
)
