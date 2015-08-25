CC := gcc
CC_FLAGS := -O3 -fPIC -Wall -Werror
LD_FLAGS := -shared -fPIC

#INCLUDES := -I/usr/local/cuda/include
INCLUDES := 

SO_TARGET := libcudatrace.so
SO_OBJECTS := lib.o

.PHONY: all clean

%.o: %.c
	$(CC) $(CC_FLAGS) $(INCLUDES) -o $@ -c $<

$(SO_TARGET): $(SO_OBJECTS)
	$(CC) $(LD_FLAGS) -o $@ $^
