#include <iostream>
#include <sqlite3.h>

int main() {
    sqlite3* db;
    int rc = sqlite3_open("data/my_database.db", &db);

    if (rc) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        return rc;
    } else {
        std::cout << "Database connected successfully!" << std::endl;
    }

    sqlite3_close(db);
    return 0;

}
