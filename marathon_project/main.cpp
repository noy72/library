#include <bits/stdc++.h>

#define range(i, a, b) for (int i = (a); i < (b); i++)
#define rep(i, b) for (int i = 0; i < (b); i++)
#define all(a) (a).begin(), (a).end()
#define show(x) cerr << #x << " = " << (x) << endl;

using namespace std;
using namespace std::chrono;

template<typename X, typename T>
auto vectors(X x, T a) {
    return vector<T>(x, a);
}

template<typename X, typename Y, typename Z, typename... Zs>
auto vectors(X x, Y y, Z z, Zs... zs) {
    auto cont = vectors(y, z, zs...);
    return vector<decltype(cont)>(x, cont);
}

template<typename T>
ostream &operator<<(ostream &os, vector<T> &v) {
    rep(i, v.size()) { os << v[i] << (i == v.size() - 1 ? "" : " "); }
    return os;
}

template<typename T, typename S>
ostream &operator<<(ostream &os, pair<T, S> &p) {
    os << '(' << p.first << ',' << p.second << ')';
    return os;
}

template<typename T>
istream &operator>>(istream &is, vector<T> &v) {
    for (T &x : v) { is >> x; }
    return is;
}

class Solver {
private:

public:
    void init() {

    }

    void read(char *name = nullptr) {
        FILE *fp = name == nullptr ? stdin : fopen(name, "r");

        // fscanf(fp, "%d", &n);

        fclose(fp);
    }

    void solve() {

    }

    long long calScore() { // 出力に対する得点を計算して返す
        return -1;
    }

    void output() {

    }
};

class Result {
    char res[64];
public:
    long long score, time;
    int testcase;

    Result() : score(0), time(0) {}

    Result(long long score, long long time, int testcase) : score(score), time(time), testcase(testcase) {}

    string to_string() {
        sprintf(res, "%3d  %12lld  %5lld msec\n", testcase, score, time);
        return string(res);
    }

    static bool rate_comp(Result &a, Result &b) { return a.score < b.score; }

    static bool score_comp(Result &a, Result &b) { return a.score < b.score; }

};


// Runner -------------------------------------------------------------------------------------
mutex mtx_;
char _name[32];
int test_size;
int thread_size;
const int progress_bar_size = 40;
string progress;
vector<Result> res;

void __addResult(Result result) {
    lock_guard<std::mutex> lock(mtx_);
    res.emplace_back(result);
}

void __showProgress() {
    rep(i, res.size() * progress_bar_size / test_size) progress[i] = '=';
    fprintf(stderr, "\r[%s] %3lu / %3d", progress.c_str(), res.size(), test_size);
}

void __runTests(int a, int b) {
    Solver s;
    range(i, a, b) {
        auto start_time = high_resolution_clock::now();

        s.init();
        sprintf(_name, "testcases/testcase-%d.in", i);
        s.read(_name);
        s.solve();

        auto dif = high_resolution_clock::now() - start_time;
        auto cal_time = duration_cast<milliseconds>(dif).count();

        __addResult(Result(s.calScore(), cal_time, i));
        __showProgress();
    }

}

inline void __output(const string &tag, Result &r) {
    fprintf(stderr, "%s: %s\n", tag.c_str(), r.to_string().c_str());
}

void __output(const string &title, function<bool(Result &, Result &)> comp) {
    fprintf(stderr, "%s", title.c_str());
    sort(all(res), comp);
    __output("max", res.back());
    __output("min", res.front());
    __output("mid", res[res.size() / 2]);
    fprintf(stderr, "\n");
}

void runner_init(int test_s, int thread_s) {
    test_size = test_s;
    thread_size = thread_s;
    rep(i, progress_bar_size) progress += ' ';
}

void runner_run(int test_s, int thread_s) {
    runner_init(test_s, thread_s);

    vector<int> s{0};
    while (s.back() < test_size) {
        s.emplace_back(s.back() + test_size / thread_size);
        if (s.back() > test_size) s.back() = test_size;
    }

    vector<thread> threads;
    rep(i, s.size() - 1) {
        threads.emplace_back(thread(__runTests, s[i], s[i + 1]));
    }
    for (auto &t : threads) t.join();
    //cerr << endl;

    __output("Score", Result::score_comp);
    __output("Score Rate", Result::rate_comp);

    fprintf(stderr, "\nAverage\n");
    Result sum;
    for (auto r : res) {
        sum.score += r.score;
        sum.time += r.time;
    }
    sum.score /= test_size;
    sum.time /= test_size;
    sum.testcase = -1;

    __output("ave", sum);

    //rep(i,test_size){show(res[i].to_string();}
}

// Runner -------------------------------------------------------------------------------------

signed main(int args, char **argv) {
    if (args == 2) {
        cerr << "[*] start" << endl;
        runner_run(100, 4);
        return 0;
    }

    Solver s;
    cerr << "[*] input testcase" << endl;
    s.read();
    s.solve();
    show(s.calScore())
    //s.output();
}
