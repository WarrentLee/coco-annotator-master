FROM nanpelee/pytorch:maskrcnn-v2.1

COPY ./torch_env/maskrcnn-benchmark /MaskrcnnPytorch/maskrcnn-benchmark

#Build project
RUN cd maskrcnn-benchmark && python setup.py build develop

ENTRYPOINT ["bash", "-c", "jupyter notebook --allow-root -y --no-browser --config=jupyter_config.py"]

EXPOSE 50

CMD  ["/bin/bash"]