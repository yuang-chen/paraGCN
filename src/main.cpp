#include "parser.h"
#include "gcn.h"
#include "variable.h"

using namespace std;

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    if(argc < 2) {
        cout << "gcn-* graph_dataset" << endl;
        return EXIT_FAILURE;
    }

    GCNParams params = GCNParams::get_default();
    GCNData data;
    string input_name(argv[1]);
    Parser parser(&params, &data, input_name);
    if(!parser.parse()) {
        cerr << "cannot read input: " << input_name << endl;
        exit(EXIT_FAILURE);
    }

    //GCN gcn(params, &data);
    Variable v(10);
    v.print();
    v.glorot(2,5);
    v.print();
    return EXIT_SUCCESS;
}