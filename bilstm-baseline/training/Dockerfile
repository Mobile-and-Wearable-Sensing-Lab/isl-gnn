FROM ubuntu:24.10@sha256:eea047b4b181f2d3aeafbc0ce5294a2bbb3b98153a68b9ed4bc573d871ca9450
# FROM ubuntu:22.04@sha256:0e5e4a57c2499249aafc3b40fcd541e9a456aab7296681a3994d631587203f97
ENV DEBIAN_FRONTEND=nointeractive
WORKDIR /service
RUN apt update && apt install -y python3 python3-pip python3-venv
RUN apt update && apt install -y ffmpeg libsm6 libxext6
COPY ./requirements.txt ./
RUN pip install --break-system-packages -r requirements.txt
# RUN pip install -r requirements.txt
COPY ./ ./
# ENTRYPOINT ["python3", "/service/main.py"]
# CMD ["hands", "--help"]
CMD ["python3", "train.py"]
# CMD ["python3", "static_runner.py"]