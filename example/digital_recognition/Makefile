MACRO=DEBUGALL
CFLAGS+=-pg -w -D$(MACRO)
	SRCPATH=.
	INCLUDEPATH=. ../../matrix ../../layers ../../utils ../../net
	TARGET=test_mnist
SOURCES=$(wildcard $(SRCPATH)/*.cpp)

#OBJS := $(patsubst %.cpp, %.o,$(SOURCES))
OBJS=$(SOURCES:.cpp=.o)
ALIAS_O=$(wildcard ../../layers/*.o)
CFLAGS+=$(foreach dir, $(INCLUDEPATH), -I$(dir))

all: build
	@echo "Build Done!"
	@echo $(CFLAGS)
clean:
	$(RM) $(OBJS) $(TARGET)

build:
	g++ -c ${CFLAGS} ${SOURCES}
	g++ ${CFLAGS} -o ${TARGET} ${OBJS} ${ALIAS_O}
