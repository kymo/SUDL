MACRO=DEBUGALL 
CFLAGS+=-w -g -std=c++11
	SRCPATH=.
	INCLUDEPATH=. ../matrix
	TARGET=test_conv_layer 
SOURCES=$(wildcard $(SRCPATH)/*.cpp)

#OBJS := $(patsubst %.cpp, %.o,$(SOURCES))
OBJS=$(SOURCES:.cpp=.o)
CFLAGS+=$(foreach dir, $(INCLUDEPATH), -I$(dir))

all: build
	@echo "Build Done!"

clean:
	$(RM) $(OBJS) $(TARGET)

build:
	g++ -c ${CFLAGS} ${SOURCES}
	g++ ${CFLAGS} -o ${TARGET} ${OBJS}
