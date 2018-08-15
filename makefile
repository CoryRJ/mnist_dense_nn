EXEFILE = myprogram

main.o: main.cpp
	g++ -Wall main.cpp TRI/to_read.cpp

clean:
	rm $(EXEFILE) main.o
