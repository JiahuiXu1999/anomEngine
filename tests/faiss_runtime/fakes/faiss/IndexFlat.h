#pragma once
#include "faiss/Index.h"
namespace faiss {
struct IndexFlatL2 : Index { explicit IndexFlatL2(int dimension) : Index(dimension) {} };
}
