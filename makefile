EXEFILE = myprogram

main.o: main.cpp
	g++ -Wall -fno-stack-protector main.cpp TRI/to_read.cpp nn_class/Dnn.cpp

clean:
	rm $(EXEFILE) main.o
