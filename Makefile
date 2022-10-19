TRANSFORMATIONS:=project perspective equidistant cylindrical-equidistant

define test_identity_transformation
tmp/SIPI_Jelly_Beans_4.1.07_$t.png: test/SIPI_Jelly_Beans_4.1.07.jpg transform.py Makefile
	mkdir --parent tmp
	python3 transform.py --input test/SIPI_Jelly_Beans_4.1.07.jpg --srctransform $t --transform $t --output "out=np.round(img).astype(np.uint8)" tmp/{}_$t.png png

test_$t: test/compare.py tmp/SIPI_Jelly_Beans_4.1.07_$t.png
	python3 test/compare.py test/SIPI_Jelly_Beans_4.1.07.jpg tmp/SIPI_Jelly_Beans_4.1.07_$t.png
test_all: test_$t
endef
$(foreach t, $(TRANSFORMATIONS), $(eval $(call test_identity_transformation)))
