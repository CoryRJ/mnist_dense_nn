EXEFILE = myprogram

main.o: main.cpp
	g++ -Wall main.cpp TRI/to_read.cpp nn_class/Cnn.cpp

clean:
	rm $(EXEFILE) main.o
