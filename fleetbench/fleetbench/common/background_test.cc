#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>
#include <cstdlib>

void RunBackgroundCommand(std::filesystem::path file, std::filesystem::path lock_file) {
    std::ifstream infile(file);
    std::string command;
    if (std::getline(infile, command)) {
        std::string modified_command = "LOCK_FILE_PATH=" + lock_file.string() + " " + command;
        std::cerr << "Running command: " << modified_command << std::endl;
        std::thread([modified_command]() {
            int ret_code = std::system(modified_command.c_str());
            if (ret_code != 0) {
                std::cerr << "Command failed with return code: " << ret_code << std::endl;
            }
        }).detach();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <command_file> <lock_file>" << std::endl;
        return 1;
    }

    std::filesystem::path command_file = argv[1];
    std::filesystem::path lock_file = argv[2];

    RunBackgroundCommand(command_file, lock_file);

    // Keep the main thread alive to allow the background thread to complete
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
