FROM nanpelee/pytorch:maskrcnn-v2.0

COPY ./torch_models/apex /MaskrcnnPytorch/apex
COPY ./torch_models/cocoapi /MaskrcnnPytorch/cocoapi
COPY ./torch_models/maskrcnn-benchmark /MaskrcnnPytorch/maskrcnn-benchmark

#Build project
RUN cd cocoapi/PythonAPI && python setup.py build_ext install \
    && cd ../../apex && python setup.py install --cuda_ext --cpp_ext \
    && cd ../maskrcnn-benchmark && python setup.py build develop

ENTRYPOINT ["bash", "-c", "jupyter notebook --allow-root -y --no-browser --config=jupyter_config.py"]

EXPOSE 50