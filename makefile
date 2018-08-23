EXEFILE = myprogram

main.o: main.cpp
	g++ -Wall -fno-stack-protector main.cpp TRI/to_read.cpp nn_class/Cnn.cpp

clean:
	rm $(EXEFILE) main.o
