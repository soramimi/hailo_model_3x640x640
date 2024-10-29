all:

onnx:
	python makeonnx.py

hef:
	python makehef.py

clean:
	-rm *.log
	-rm *.hef
	-rm *.onnx
	-rm *.har
