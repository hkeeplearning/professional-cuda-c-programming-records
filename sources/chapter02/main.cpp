#include <stdio.h>

extern "C" void HelloWorld();

int main()
{
    printf("Hello world from CPU!\n");
    HelloWorld();
    return 0;
}