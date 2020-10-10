all:
	mkdir -fp sample/masks
	wget --https-only -o "sample/masks/?.png" http://freevector.co/wp-content/uploads/2010/10/37171-question-mark-in-a-circle-outline.png
	#wget --https-only -o "sample/lorem/message.txt" https://tools.wmflabs.org/wsexport/tool/book.php?lang=la&format=txt&page=Lorem_ipsum
	../src/piet_modeler.py "sample/masks/?.png" --output doc/lorem.png -f sample/lorem/message.txt
