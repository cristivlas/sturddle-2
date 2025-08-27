#include <iostream>
#include <stdexcept>
#include "context.h"
#include <sanitizer/msan_interface.h>

#define VERSION "2.3 Native " _TOSTR(BUILD_STAMP)


std::unordered_map<std::string, std::string> make_params()
{
    std::unordered_map<std::string, std::string> params;
    params.emplace(std::string("name"), std::string("Sturddle"));
    params.emplace(std::string("version"), std::string(VERSION));

    return params;
}



int main()
{
    try
    {
        search::Context::init();
        uci_loop(make_params());
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

