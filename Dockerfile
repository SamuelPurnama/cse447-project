FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Copy and install dependencies from requirements.txt
COPY requirements.txt /job/
RUN pip install -r requirements.txt

# You can also add additional packages directly here if needed:
# RUN pip install tqdm
