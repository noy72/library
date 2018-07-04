#rightHand
# 右手法
#dijkstra-extend
# 拡張dijkstra
#grundy
# グランディー数を求める再帰関数
#eulerGraph
# オイラー路の構築
#AhoCorasick
# 文字列を扱う木構造
#bitSizeSet
# 0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
#bitSubSet
# 部分集合のビット列を降順に列挙する
#nibu-graph
# 二部グラフ判定・色塗り
#levenshteinDistance
# 編集距離
#matrixChainMultiplication
# 連鎖行列積
#maximumrectangule
# 最大長方形
#cumulativeSum2D
# 2次元累積和
#maximumSubarray
# 連続した部分列の和の最大値を求める
#treeDP
# 全方位木DPによる木の直径の演算
#closedLoop
# 閉路の検出
#JoinInterval
# 区間の結合
#repalceAll
# 文字列の置き換え
#eulerPhi
# オイラー関数
#extgcd
# 拡張ユークリッドの互除法
#simultaneousLinearEquations
# 連立一次方程式
#flow
# 最大流
#matrix
# 行列計算
#meet_in_the_middle
# bitによる全通列挙
#compress coordinate
# 座標圧縮
#gridUnion-find
# グリッドグラフのユニオン木
#dp_Partial_sum_with_number_restriction
# 個数制限付き部分和
#pascals_triangle
# n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
#intervalState
# 区間の関係
#toDAG
# 2点間の最短経路になる辺を残したDAG
#ternarySearch
# 三分探索
#numOfnumber
# nまでの数字を書いたとき、1が出現する回数
#next_combination
# 組み合わせの全列挙
#eightQueensProblem
# 8クイーン問題
#numPrimeFactor
# 約数の個数を求める
#bitManip
# ビットの操作
#oneStrokePath
# 全ての頂点を通る一筆書きの総数
#levenshteinDistance
# 編集距離を求めるdp
#bfsOfAdjacencyMatrix
# 隣接行列の幅優先探索
#bfsOfGrid
# グリッドの幅優先探索
#dfsOfTree
# 木の深さ優先探索
#binarySearch
# 二分探索
#eratosthenes
# 10^6以下の素数を全列挙
#GreatestCommonDivisor
# 最大公約数
#gcd
# 最小公倍数
#lcm
# うるう年判定
#maze
# 迷路をグリッドに拡張
#nextMonth
# 日付の計算
#parser
# 構文解析
#power
# 冪乗
#rotate
# 行列の回転
#toNum
# 文字列から数値への変換
#toStr
# 数値から文字列への変換
#warshallFloyd
# 全点最短経路
#bellmanFord
# 負の経路を含む最短経路、負の経路の検出
#prim
# 最小無向全域木
#LCS
# 最長共通部分文字列の長さ
#LIS
# 最長増加部分列
#divisor
# 約数の列挙
#primeVactor
# 素因数分解
#syakutori
# しゃくとり
#isUniqueStr
# 文字列の文字がユニークか判定
#areAnagram
# 文字列がアナグラムかを判定
#spilt
# 文字列を空白で区切る
#topologicalSort
# トポロジカルソート
#heightOfTree
# 木の高さを求める
#bipartiteMatching
# 2部マッチング



#幅優先探索
snippet bfsOfAdjacencyMatrix
abbr 隣接行列の幅優先探索

 const int N = ;

 int M[N][N];

 void bfs(int n){
    int dis[N]; //距離
    queue<int> q; //訪問した点を入れる
    rep(i,N) dis[i] = INF;

    dis[1] = 0;
    q.push(1);

    int u;
    while(!q.empty()){
        u = q.front(); q.pop();
        rep(v,n + 1){
            if(M[u][v] && dis[v] == INF){
                dis[v] = dis[u] + 1; //グラフの深さ 
                q.push(v);
            }
        }
    }
 }

snippet     bfsOfGrid
abbr        グリッドの幅優先探索

 const int N;
 
 struct Point{ int x, y; };
 int dy[4] = {0,1,0,-1}, dx[4] = {1,0,-1,0};
 bool M[N][N];
 
 int bfs(int h, int w, Point p){
     int dis[N][N];
     queue<Point> q;
     rep(i,N) rep(j,N) dis[i][j] = INF;
 
     dis[p.y][p.x] = 0;
     q.push(p);
 
     Point u;
     while(not q.empty()){
         u = q.front(); q.pop();
         rep(i,4){
             Point next;
             next.x = u.x + dx[i];
             next.y = u.y + dy[i];
             if(next.x < 0 || next.x >= w || next.y < 0 || next.y >= h) continue;
             if(dis[next.y][next.x] == INF && M[next.y][next.x]){
                 dis[next.y][next.x] = dis[u.y][u.x] + 1;
                 q.push(next);
             }
         }
     }
     return /*返り値*/;
 }


snippet     dfsOfTree
abbr        木構造の深さ優先探索
options     head

 typedef struct{
     int parent, left, right;
 } Node;
 Node t[1002];
 
 void dfs(int u, int d){
     if(t[u].left != INF){
         dfs(m, t[u].left, d + 1);
     }
     if(t[u].right != INF){
         dfs(m, t[u].right, d);
     }
 }
 
 int brotherNum(int u){
     if(t[u].right == INF){
         return u;
     }else{
         return brotherNum(t[u].right);
     }
 }
 
 void inputData(int par){
     t[i + 1].parent = par;
     if(t[par].left == INF){
         t[par].left = i + 1;
     }else{
         t[brotherNum(t[par].left)].right = i + 1;
     }
 }
 
 void printGraph(){
     range(i,1,n + 1){ cout << t[i].parent << ' ' << t[i].left << ' ' << t[i].right << endl; }
 }

snippet     binarySearch
abbr        二分探索

 int right = , left = ;
 rep(i,100){
     int mid = (right + left) / 2;
     if(C(mid)) right = mid;
     else left = mid;
 }

snippet     eratosthenes
abbr        10^6以下の素数を全列挙
options     head
    
 const int kN;
 void primeNumber(bool prime[kN]){
     rep(i,kN) prime[i] = 1;
     prime[0] = prime[1] = 0;
     rep(i,kN){
         if(prime[i]){
             for(int j = i + i; j < kN; j+=i){
                 prime[j] = 0;
             }
         }
     }
 }

snippet     gcd
abbr        最大公約数
options     head
    
 int gcd(int x, int y) {
    int r;
    if(x < y) swap(x, y);

    while(y > 0){
        r = x % y;
        x = y;
        y = r;
    }
    return x;
 }

snippet     lcm
abbr        最小公倍数
options     head

 int gcd(int x, int y) {
    int r;
    if(x < y) swap(x, y);

    while(y > 0){
        r = x % y;
        x = y;
        y = r;
    }
    return x;
 }
    
 int lcm( int m, int n ) {
    // 引数に０がある場合は０を返す
    if ( ( 0 == m ) || ( 0 == n ) ) return 0;
    return ((m / gcd(m, n)) * n); // lcm = m * n / gcd(m,n)
 }

snippet     isPrime
abbr        素数判定
options     head
    
 bool primeNumber(int n){
    if(n < 2) return 0;
    else{
        for(int i = 2; i * i <= n; i++){
            if(n % i == 0) return 0;
        }
        return 1;
    }
 }

snippet     leapYear
abbr        うるう年判定
options     head
    
 /*
 うるう年判定
 400で割り切れる年数から数えて、0-399年間でうるう年は97回
 */
 bool leapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }

snippet     maze
abbr        迷路を二次平面上に拡張する
options     head
    
 const int N = ;

 void printMaze(int w, int h, bool M[N][N]){
    rep(i,h + h - 1){
        rep(j,w + w + 1){
            cout << M[i][j];
        }
        cout << endl;
    }
 }

 void extensionOfMaze(int w, int h, bool M[N][N]){
    rep(i,N) rep(j,N) M[i][j] = 1;
    rep(i,h + h - 1){
        if(i % 2){ //横線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 0) M[i + 1][j + 1] = 1;
                else cin >> M[i + 1][j + 1];
            }
        }else{ //縦線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 1) M[i + 1][j + 1] = 0;
                else cin >> M[i + 1][j + 1];
            }
        }
    }
 }

snippet     nextMonth
abbr        日付の計算
options     head

 bool isLeapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }
    
 void nextMonth(int &y, int &m, int &d){
    bool leapYear = isLeapYear(y);
    if((m == 2 && d == 30 && leapYear) ||
       (m == 2 && d == 29 && !leapYear) ||
       ((m == 4 || m == 6 || m == 9 || m == 11) && d == 31) ||
       ((m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) && d == 32)){
        m++;
        d = 1;
    }
    if(m == 13){
        y++;
        m = 1;
    }
 }

snippet     parser
abbr        構文解析
options     head
    
 typedef string::const_iterator State;
 int number(State&);
 int factor(State&);
 int term(State&);
 int expression(State&);
 
 // 数字の列をパースして、その数を返す。
 int number(State &begin) {
     int ret = 0;
 
     while (isdigit(*begin)) {
         ret *= 10;
         ret += *begin - '0';
         begin++;
     }
 
     return ret;
 }
 
 // 括弧か数をパースして、その評価結果を返す。
 int factor(State &begin) {
     if (*begin == '(') {
         begin++; // '('を飛ばす。
         int ret = expression(begin);
         begin++; // ')'を飛ばす。
         return ret;
     } else {
         return number(begin);
     }
     return 0;
 }
 
 // 乗算除算の式をパースして、その評価結果を返す。
 int term(State &begin) {
     int ret = factor(begin);
 
     for (;;) {
         if (*begin == '*') {
             begin++;
             ret *= factor(begin);
         } else if (*begin == '/') {
             begin++;
             ret /= factor(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }
 
 // 四則演算の式をパースして、その評価結果を返す。
 int expression(State &begin) {
     int ret = term(begin);
 
     for (;;) {
         if (*begin == '+') {
             begin++;
             ret += term(begin);
         } else if (*begin == '-') {
             begin++;
             ret -= term(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }

 //beginがexpectedを指していたらbeginを一つ進める。
 void consume(State &begin, char expected) {
     if (*begin == expected) {
         begin++;
     } else {
         cerr << "Expected '" << expected << "' but got '" << *begin << "'" << endl;
         cerr << "Rest string is '";
         while (*begin) {
             cerr << *begin++;
         }
         cerr << "'" << endl;
         //throw ParseError();
     }
 }

snippet     power
abbr        冪乗
options     head

 //x^n mod M
 typedef unsigned long long ull;
 const ull M = 1000000007;
 ull power(ull x, ull n){
     ull res = 1;
     if(n > 0){
         res = power(x, n / 2);
         if(n % 2 == 0) res = (res * res) % M;
         else res = (((res * res) % M) * x ) % M;
     }
     return res;
 }

snippet     rotate
abbr        行列の回転
options     head
 //P is structure of coodinate.
 void rotationMatrix(P &p, double angle){
     double x, y;
         x = p.x * cos(angle) - p.y * sin(angle);
         y = p.x * sin(angle) + p.y * cos(angle);
         p.x = x;
         p.y = y;
 }

snippet     toNum
abbr        文字列から数値への変換
options     head
    
 //文字列から数値への変換
 int toNum(string str){
    int num = 0;
    rep(i,str.size()){
        num *= 10;
        num += str[i] - '0';
    }
    return num;
 }

snippet     toStr
abbr        数値から文字列への変換
options     head
    
 string toStr(int n){
    string str;
    int len = static_cast<int>(log10(n));
    int K = 1;
    rep(i,len) K*=10;
    rep(i,len + 1){
        if(n / K == 0) str+= '0';
        else str+= ('0' + n / K);
        n%=K; K/=10;
    }
    return str;
 }

snippet     warshallFloyd
abbr        全点最短経路
options     head

 const int MAX_V = ;

 void init(int m[MAX_V][MAX_V]){
     rep(i,MAX_V) rep(j,MAX_V) m[i][j] = INF;
     rep(i,MAX_V) m[i][i] = 0;
 }

 void warshallFloyd(int m[MAX_V][MAX_V], int n){
     rep(k,n){
         rep(i,n){
             rep(j,n){
                 m[i][j] = min(m[i][j], m[i][k] + m[k][j]);
             }
         }
     }
 }


snippet     bellmanFord
abbr        負の経路を含む最短経路、負の経路の検出
options     head

 class Edge{
     public:
         int to, cost;
         Edge(int to, int cost) : to(to) ,cost(cost) {}
 };
 
 typedef vector<vector<Edge>> AdjList;
 vector<int> dis;
 
 bool bellmanFord(AdjList g, int n, int s) { // nは頂点数、sは開始頂点
     dis = vector<int>(n, INF);
     dis[s] = 0; // 開始点の距離は0
     rep(i,n){
         rep(v,n){
             rep(k,g[v].size()){
                 Edge e = g[v][k];
                 if (dis[v] != INF && dis[e.to] > dis[v] + e.cost) {
                     dis[e.to] = dis[v] + e.cost;
                     if (i == n - 1) return true; // n回目にも更新があるなら負の閉路が存在
                 }
             }
         }
     }
     return false;
 }
 
snippet     prim
abbr        最小全域木
options     head

 class Edge {
     public:
     int to, cost;
     Edge(int to, int cost) : to(to), cost(cost) { }
 };
 
 typedef vector<Edge> Edges;
 typedef vector<Edges> Graph;
 
 int prim(const Graph &g, int s = 0) {
     int n = g.size();
     int total = 0;
 
     vector<bool> visited(n);
     //priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
     priority_queue<pair<int, int> > q;
 
     q.push(make_pair(0,s));
     while (not q.empty()) {
         pair<int, int> u = q.top(); q.pop();
         if (visited[u.second]) continue;
         total += u.first;
         visited[u.second] = true;
         for(auto it : g[u.second]) {
             if (not visited[it.to]) q.push(make_pair(it.cost, it.to));
         }
     }
     return total;
 }

snippet     LCS
abbr        最長共通部分文字列の長さ
options     head

 const int MAX_N = 1000;
 
 int solve(string s, string t){
     int dp[MAX_N + 1][MAX_N + 1] = {{0}};

     rep(i,s.size()){
         rep(j,t.size()){
             if(s[i] == t[j]){
                 dp[i + 1][j + 1] = dp[i][j] + 1;
             }else{ //連続した部分文字列の場合はここを消し、dpのmaxを返す
                 dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
             }
         }
     }
     return dp[n][m];
 }

snippet     LIS
abbr        最長増加部分列
options     head
 const int MAX_N = 1000;
 int dp[MAX_N];

 int LIS(int n, int a[MAX_N]){
     fill(dp, dp + n, INF);
     rep(i,n){
         *lower_bound(dp, dp + n, a[i]) = a[i];
     }
     return lower_bound(dp, dp + n, INF) - dp;
 }

snippet     divisor
abbr        約数の列挙
options     head
 vector<int> divisor(int n){
     vector<int> res;
     for(int i = 1; i * i <= n; i++){
         if(n % i == 0){
             res.emplace_back(i);
             if(i != n / i) res.emplace_back(n / i);
         }
     }
     return res;
 }

snippet     primeFactor
abbr        素因数分解
options     head
 map<int, int> primeFactor(int n){
     map<int, int> res;
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] = 1;
     return res;
 }

snippet     syakutori
abbr        しゃくとり
options     head
 //S:sumの条件 n:個数
 int solve(){
     int res = n + 1;
     int s = 0, t = 0, sum = 0;
     while(true){
         while(t < n && sum < S){
             sum += a[t++];
         }
         if(sum < S) break;
         res = min(res, t - s);
         sum -= a[s++];
     }
     if(res > n){
         res = 0;
     }
     return res;
 }

snippet     isUniqueStr
abbr        文字列の文字がユニークか判定
options     head
    
 bool isUniqueStr(string s){
     unsigned int char_set[128] = {0};
     rep(i,s.size()) char_set[s[i]]++;
     rep(i,128) if(char_set[i] >= 2) return 0;
     return 1;
 }

snippet     areAnagram
abbr        文字列がアナグラムかを判定
options     head

 bool areAnagram(string s, string t){
     if(s.size() != t.size()) return 0;
 
     unsigned int char_set[128] = {0};
     rep(i,s.size()){
         char_set[s[i]]++;
         char_set[t[i]]--;
     }
     rep(i,128) if(char_set[i] != 0) return 0;
     return 1;
 }

snippet     split
abbr        文字列を空白で区切る
options     head

 vector<string> split(string in, char sp = ' '){
     vector<string> ret;
     stringstream ss(in);
     string s;
     while(getline(ss, s, sp)){
         ret.emplace_back(s);
     }
     return ret;
 }


snippet     topologicalSort
abbr        トポロジカルソート
options     head

 const int MAX_V = 10000;
 
 vector<int> g[MAX_V]; //グラフ
 vector<int> tp; //トポロジカルソートの結果
 
 void bfs(int s, int indeg[MAX_V], bool used[MAX_V]){
     queue<int> q;
 
     q.push(s);
     used[s] = true;
 
     while(not q.empty()){
         int u = q.front(); q.pop();
         tp.emplace_back(u);
         rep(i,g[u].size()){
             int v = g[u][i];
             indeg[v]--;
             if(indeg[v] == 0 && not used[v]){
                 used[v] = true;
                 q.push(v);
             }
         }
     }
 }
 
 //グラフに閉路がある場合、0を返す
 bool topologicalSort(int v){
     int indeg[MAX_V]; //入次数
     bool used[MAX_V];
     memset(indeg, 0, sizeof(indeg));
     memset(used, 0, sizeof(used));
 
     rep(i,v) rep(j,g[i].size()) indeg[ g[i][j] ]++;
     rep(i,v) if(indeg[i] == 0 && not used[i]) bfs(i, indeg, used);
 
     for(auto it:tp) cout << it << endl;
 
     if(tp.size() == v) return true;
     else return false;
 }

snippet     heightOfTree
abbr        木の高さを求める
options     head
    
 const int MAX_V = 10000;
 
 class Edge{
     public:
         int dst, weight;
         Edge(){}
         Edge(int dst, int weight): dst(dst), weight(weight)  {}
 };
 
 typedef vector<vector<Edge>> Graph;
 
 Graph g(MAX_V);
 
 int visit(Graph &t, int i, int j) {
     if(t[i][j].weight >= 0) return t[i][j].weight;
     t[i][j].weight = g[i][j].weight;
     int u = t[i][j].dst;
     rep(k,t[u].size()) {
         if(t[u][k].dst == i) continue;
         t[i][j].weight = max(t[i][j].weight, visit(t,u,k) + g[i][j].weight);
     }
     return t[i][j].weight;
 }
 vector<int> height(int n) {
     Graph t = g;
     rep(i,n) rep(j,t[i].size()) t[i][j].weight = -1;
     rep(i,n) rep(j,t[i].size()) if(t[i][j].weight < 0) t[i][j].weight = visit(t, i, j);
 
     vector<int> ht(n); // gather results
     rep(i,n) rep(j,t[i].size()) ht[i] = max(ht[i], t[i][j].weight);
     return ht;
 }



snippet     bipartiteMatching
abbr        2部マッチング
options     head

 const int MAX_V = 210; //MAX_X + MAX_Y
 const int MAX_X = 105;
 const int MAX_Y = 105;
 
 class Edge{
     public:
         int to, cap, rev;
 };
 
 typedef vector<vector<Edge>> AdjList;
 AdjList G(MAX_V);
 bool used[MAX_V];
 
 void addEdge(int from, int to, int cap){
     G[from].emplace_back(Edge{to, cap, static_cast<int>(G[to].size())});
     G[to].emplace_back(Edge{from, 0, static_cast<int>(G[from].size() - 1)});
 }
 
 int dfs(int v, int t, int f){
     if(v == t) return f;
     used[v] = true;
     rep(i,G[v].size()){
         Edge &e = G[v][i];
         if(not used[e.to] && e.cap > 0){
             int d = dfs(e.to, t, min(f, e.cap));
             if(d > 0){
                 e.cap -= d;
                 G[e.to][e.rev].cap += d;
                 return d;
             }
         }
     }
     return 0;
 }
 
 int maxFlow(int s, int t){
     int flow = 0;
     while(true){
         memset(used, 0, sizeof(used));
         int f = dfs(s, t, INF);
         if(f == 0) return flow;
         flow += f;
     }
 }
 
 int bipartiteMatching(int x, int y, bool edge[MAX_X][MAX_Y]){
     int s = x + y, t = s + 1; //set x : 0 ~ x-1, set y : x ~ x+y-1
 
     rep(i,x) addEdge(s, i, 1); //sと集合xを結ぶ
     rep(i,y) addEdge(x + i, t, 1); //集合yとtを結ぶ
 
     rep(i,x) rep(j,y) if(edge[i][j]) addEdge(i, x + j, 1); //集合xと集合yを結ぶ
 
     return maxFlow(s, t);
 }

snippet     levenshteinDistance
abbr        編集距離を求めるdp
 const int MAX_N = 1005;
 int dp[MAX_N][MAX_N];
 
 int minimum(int a, int b, int c){
     return min(min(a,b),c);
 }
 
 int levenshteinDistance(string a, string b){
     rep(i,a.size() + 1) dp[i][0] = i;
     rep(i,b.size() + 1) dp[0][i] = i;
 
     range(i,1,a.size() + 1){
         range(j,1,b.size() + 1){
             int cost = a[i - 1] == b[j - 1] ? 0 : 1;
             dp[i][j] = minimum(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
         }
     }
     return dp[a.size()][b.size()];
 }

snippet     oneStrokePath
abbr        全ての頂点を通る一筆書きの総数
 const int MAX_V = 10;
 
 int n;
 int ans = 0;
 bool M[MAX_V][MAX_V] = {0};
 
 void dfs(int c, int u, vector<bool> used){
     if(u == n - 1){
         ans++;
         return;
     }
     rep(i,n){
         if(M[c][i] == 1 && used[i] == false){
             used[c] = true;
             dfs(i,u + 1,used);
         }
     }
 }

snippet     bitManip
abbr        ビットの操作
 //i番目のビットを返す
 bool getBit(int num, int i){
     return ((num & (1 << i)) != 0);
 }
 
 //i番目を1にする
 int setBit(int num, int i){
     return num | (1 << i);
 }
 
 //i番目を0にする
 int clearBit(int num, int i){
     int mask = ~(1 << i);
     return num & mask;
 }
 
 //i番目をvで置き換える
 int updateBit(int num, int i, int v){
     int mask = ~(1 << i);
     return (num & mask) | (v << i);
 }

snippet     numPrimeFactor
abbr        約数の個数を求める
 const long long M = 1000000007;
 map<int, int> res;
 
 void primeFactor(int n){
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] += 1;
 }
 
 long long numPrimeFactor(){
     long long ans = 1;
     for(auto it:res){
         ans = ans * (it.second + 1);
         ans %= M;
     }
     return ans;
 }

snippet     eightQueensProblem
abbr        8クイーン問題
 const int N = 8;
 
 //row[i] = j : (i,j)にクイーン
 int row[N], col[N], dpos[2 * N - 1], dneg[2 * N - 1];
 char x[8][8];
 bool f = false;
 
 void init(){
     rep(i,N){
         row[i] = 0;
         col[i] = 0;
     }
     rep(i,2 * N - 1){
         dpos[i] = 0;
         dneg[i] = 0;
     }
 }
 
 void printBoard(){
     rep(i,N){
         rep(j,N){
             if(x[i][j] == 'Q'){
                 if(row[i] != j) return;
             }
         }
     }
     rep(i,N){
         rep(j,N){
             cout << ( (row[i] == j) ? "Q" : ".");
         }
         cout << endl;
     }
     f = true;
 }
 
 void recursive(int i){
     if(i == N){
         printBoard();
         return;
     }
 
     rep(j,N){
         if(col[j] || dpos[i + j] || dneg[i - j + N - 1]) continue;
         row[i] = j;
         col[j] = dpos[i + j] = dneg[i - j + N - 1] = 1;
         recursive(i + 1);
         row[i] = col[j] = dpos[i + j] = dneg[i - j + N - 1] = 0;
         if(f) return;
     }
 }

snippet     next_combination
abbr        組み合わせの全列挙
 template < class BidirectionalIterator >
 bool next_combination ( BidirectionalIterator first1 ,
         BidirectionalIterator last1 ,
         BidirectionalIterator first2 ,
         BidirectionalIterator last2 ){
     if (( first1 == last1 ) || ( first2 == last2 )) {
         return false ;
     }
     BidirectionalIterator m1 = last1 ;
     BidirectionalIterator m2 = last2 ; --m2;
     while (--m1 != first1 && !(* m1 < *m2 )){
     }
     bool result = (m1 == first1 ) && !(* first1 < *m2 );
     if (! result ) {
         while ( first2 != m2 && !(* m1 < * first2 )) {
             ++ first2 ;
         }
         first1 = m1;
         std :: iter_swap (first1 , first2 );
         ++ first1 ;
         ++ first2 ;
     }
     if (( first1 != last1 ) && ( first2 != last2 )) {
         m1 = last1 ; m2 = first2 ;
         while (( m1 != first1 ) && (m2 != last2 )) {
             std :: iter_swap (--m1 , m2 );
             ++ m2;
         }
         std :: reverse (first1 , m1 );
         std :: reverse (first1 , last1 );
         std :: reverse (m2 , last2 );
         std :: reverse (first2 , last2 );
     }
     return ! result ;
 }
 
 template < class BidirectionalIterator > bool next_combination ( BidirectionalIterator first , BidirectionalIterator middle , BidirectionalIterator last )
 {
     return next_combination (first , middle , middle , last );
 }
 
 //要素vからr個取り出す組み合わせ
 void func(vector<int> v, int r){
     do{
     }while(next_combination(v.begin(), v.begin() + r, v.end()));
 }

snippet     numOfnumber
abbr        nまでの数字を書いたとき、1が出現する回数
 //nに含まれる1の数
 long long numOfnumber(long long n){
     long long k = 10;
     long long ans = 0;
     rep(i,9){
         ans += n / k * (k / 10LL);
         ans += min(max(0LL, n % k - (k / 10LL - 1)), k / 10LL);
         k*=10LL;
     }
     return ans;
 }

snippet     ternarySearch
abbr        三分探索
 double C(double x){
 }
 
 double ternarySearch(double p){
     double right = INF, left = 0;
     rep(i,200){
         double llr = (left * 2 + right) / 3;
         double rll = (left + right * 2) / 3;
         if(C(llr) > C(rll)){
             left = llr;
         }else{
             right = rll;
         }
     }
     return left;
 }


snippet     toDAG
abbr        2点間の最短経路になる辺を残したDAG
 //出発地、出発地から全ての点へ対する最短経路、返り値、辺
 void toDAG(int s, int g[MAX_V][MAX_V], int dag[MAX_V][MAX_V], vector<pair<int, int>> v){
     rep(i,v.size()){
         if(g[s][v[i].first] + 1 == g[s][v[i].second]){
             dag[v[i].first][v[i].second] = 1;
         }
     }
 }

snippet     intervalState
abbr        区間の関係
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }

snippet     pascals_triangle
abbr        n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 //n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 void Pascals(double m[N][N]){
     m[0][0] = 1;
     range(i,1,1011){
         rep(j,i + 1){
             if(j == 0) m[i][j] = m[i - 1][j] / 2;
             else if(j == i) m[i][j] = m[i - 1][j - 1] / 2;
             else m[i][j] = (m[i - 1][j] + m[i - 1][j - 1]) / 2;
         }
     }
 }

snippet     dp_Partial_sum_with_number_restriction
abbr        個数制限付き部分和

 const int MAX_N = 105;
 const int MAX_K = 100005;

 void solve(){
     int n, k;
 
     scanf("%d%d", &n, &k);
     int a[MAX_N], m[MAX_N];
     rep(i,n) scanf("%d", &a[i]);
     rep(i,n) scanf("%d", &m[i]);
 
     int dp[MAX_K];
     memset(dp, -1, sizeof(dp));
     dp[0] = 0;
     rep(i,n){
         rep(j,k + 1){
             if(dp[j] >= 0){
                 dp[j] = m[i];
             }else if(j < a[i] || dp[j - a[i]] <= 0){
                 dp[j] = -1;
             }else{
                 dp[j] = dp[j - a[i]] - 1;
             }
         }
     }
 
     int sum = 0;
     range(i,1,k + 1){
         if(dp[i] >= 0) sum++;
     }
     cout << sum << endl;
 }

snippet     gridUnion-find
abbr        グリッドグラフのユニオン木
 const int gmax_n =1005 ;
 const int dy[3] = { 0, 1, 1};
 const int dx[3] = { 1, 0, 1};
 
 pair<int, int> par[gmax_n][gmax_n]; //親
 int depth[gmax_n][gmax_n];//木の深さ
 int cc[gmax_n][gmax_n]; //連結成分
 
 void init(int h, int w){
     rep(i,h){
         rep(j,w){
             par[i][j] = make_pair(i,j);
             depth[i][j] = 0;
             cc[i][j] = 1;
         }
     }
 }
 
 pair<int, int> find(pair<int, int> x){
     if(par[x.first][x.second] == x){
         return x;
     }else {
         return par[x.first][x.second] = find(par[x.first][x.second]);
     }
 }
 
 void unite(pair<int, int> x, pair<int, int> y){
     x = find(x);
     y = find(y);
     if(x == y) return;
 
     if(depth[x.first][x.second] < depth[y.first][y.second]){
         par[x.first][x.second] = y;
         cc[y.first][y.second] += cc[x.first][x.second];
     }else{
         par[y.first][y.second] = x;
         cc[x.first][x.second] += cc[y.first][y.second];
         if(depth[x.first][x.second] == depth[y.first][y.second]) depth[x.first][x.second]++;
     }
 }
 
 bool same(pair<int, int> x, pair<int, int> y){
     return find(x) == find(y);
 }
 
 void uniteAll(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){
         rep(j,w){
             if(m[i][j] == 'o'){
                 rep(k,3){
                     int ny = i + dy[k];
                     int nx = j + dx[k];
                     if(ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                     if(m[ny][nx] == 'o') unite(make_pair(i,j), make_pair(ny,nx));
                 }
             }
             if(i < h && j < w && m[i + 1][j] == 'o' && m[i][j + 1] == 'o'){
                 unite(make_pair(i + 1, j), make_pair(i, j + 1));
             }
         }
     }
 }
 
 void check(int n, int ans[3]){
     int i = 0;
     while(true){
         i++;
         assert(i <= 310);
         if(n % (i * i) != 0) continue;
 
         if(n / (i * i) == 12){ ans[0]++; break; }
         else if(n / (i * i) == 16){ ans[1]++; break; }
         else if(n / (i * i) == 11){ ans[2]++; break; }
     }
 }
 
 void print(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){ rep(j,w){ cout << cc[i][j] << ' '; } cout << endl; } cout << endl;
     rep(i,h){ rep(j,w){
         if(make_pair(i,j) == par[i][j]) cout << '.';
         else cout << '#';
     } cout << endl; } cout << endl;
 }


snippet     meet_in_the_middle
abbr        bitによる全通列挙
 //要素wをnとmに分け、それぞれで全列挙する
 vector<long long> a, b;
 rep(i,(1 << n)){
     long long sum = 0;
     rep(j,n){
         if(getBit(i,j)) sum += w[j];
     }
     a.emplace_back(sum);
 }
 rep(i,(1 << m)){
     long long sum = 0;
     rep(j,m){
         if(getBit(i,j)) sum += w[n + j];
     }
     b.emplace_back(sum);
 }

snippet     matrix
abbr        行列計算
 const int M = 10000;
 typedef vector<vector<int>> mat;
 
 mat mul(mat &a, mat &b){
     mat c(a.size(), vector<int>(b[0].size()));
     rep(i,a.size()){
         rep(k,b.size()){
             rep(j,b[0].size()){
                 c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % M;
             }
         }
     }
     return c;
 }
 
 mat pow(mat a, int n){
     mat b(a.size(), vector<int>(a.size()));
     rep(i,a.size()){
         b[i][i] = 1;
     }
     while(n > 0){
         if(n & 1) b = mul(b,a);
         a = mul(a, a);
         n >>= 1;
     }
     return b;
 }
 
 int solve(int n){
     mat a(2, vector<int>(2));
     a[0][0] = 1; a[0][1] = 1;//フィボナッチ数列の漸化式の行列
     a[1][0] = 1; a[1][1] = 0;
     a = pow(a,n); //行列Aのn乗。
     return a[1][0];
 }

snippet     flow
abbr        最大流
 const int MAX_V = 10005;
 
 class Edge{
     public:
         int to, cap, rev;
         Edge(int to, int cap, int rev) : to(to), cap(cap), rev(rev) {}
 };
 
 class Flow{
     private:
         vector<Edge> G[MAX_V];
         bool used[MAX_V];
         int level[MAX_V]; //sからの距離
         int iter[MAX_V]; //どこまで調べ終わったか
         int dfs(int v, int t, int f){
             if(v == t) return f;
             used[v] = true;
             rep(i,G[v].size()){
                 Edge &e = G[v][i];
                 if(not used[e.to] && e.cap > 0){
                     int d = dfs(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         int dfs_(int v, int t, int f){
             if(v == t) return f;
             for(int &i = iter[v]; i < G[v].size(); i++){
                 Edge &e = G[v][i];
                 if(e.cap > 0 && level[v] < level[e.to]){
                     int d = dfs_(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         void bfs(int s){
             memset(level, -1, sizeof(level));
             queue<int> que;
             level[s] = 0;
             que.push(s);
             while(not que.empty()){
                 int v = que.front(); que.pop();
                 rep(i,G[v].size()){
                     Edge &e = G[v][i];
                     if(e.cap > 0 && level[e.to] < 0){
                         level[e.to] = level[v] + 1;
                         que.push(e.to);
                     }
                 }
             }
         }
     public:
         void addEdge(int from, int to, int cap){
             G[from].push_back(Edge(to, cap, static_cast<int>(G[to].size())));
             G[to].push_back(Edge(from, 0, static_cast<int>(G[from].size() - 1)));
         }
         int fordFulkerson(int s, int t){
             int flow = 0;
             while(true){
                 memset(used, 0, sizeof(used));
                 int f = dfs(s, t, INF);
                 if(f == 0) return flow;
                 flow += f;
             }
         }
         int dinic(int s, int t){
             int flow = 0;
             while(true){
                 bfs(s);
                 if(level[t] < 0) return flow;
                 memset(iter, 0, sizeof(iter));
                 int f;
                 while( (f = dfs_(s, t, INF)) > 0){
                     flow += f;
                 }
             }
         }
 };

snippet     simultaneousLinearEquations
abbr        連立一次方程式
 const double EPS = 1e-8;
 typedef vector<double> vd;
 typedef vector<vd> mat;
 
 vd simultaneousLinearEquations(const mat &A, const vd &b){
     int n = A.size();
     mat B(n, vd(n + 1));
     rep(i,n) rep(j,n) B[i][j] = A[i][j];
     rep(i,n) B[i][n] = b[i];
 
     rep(i,n){
         int pivot = i;
         for(int j = i; j < n; j++){
             if(abs(B[i][j]) > abs(B[pivot][i])) pivot = j;
         }
         swap(B[i], B[pivot]);
 
         if(abs(B[i][i]) < EPS) return vd(); //解なし or 一意ではない
 
         for(int j = i + 1; j <= n; j++) B[i][j] /= B[i][i];
         rep(j,n){
             if(i != j){
                 for(int k = i + 1; k <= n; k++) B[j][k] -= B[j][i] * B[i][k];
             }
         }
     }
     vd x(n);
     rep(i,n) x[i] = B[i][n];
     return x;
 }

snippet     extgcd
abbr        拡張ユークリッドの互除法
 //ax + by = gcd(a,b) の解をもとめる
 int extgcd(int a, int b, int &x, int &y){
     int d = a;
     if(b != 0){
         d = extgcd(b, a % b, y, x);
         y -= (a / b) * x;
     }else{
         x = 1; y = 0;
     }
     return d; //gcd(x,y)
 }

snippet     eulerPhi
abbr        オイラー関数
 const int MAX_N = 100;
 
 //オイラー関数の値を求める
 int eulerPhi(int n){
     int res = n;
     for(int i = 2; i * i <= n; i++){
         if(n % i == 0){
             res = res / i * (i - 1);
             for(; n % i == 0; n /= i);
         }
     }
     if(n != 1) res = res / n * (n - 1);
     return res;
 }
 
 int euler[MAX_N];
 
 //オイラー関数のテーブルを作る
 void eulerPhi2(){
     rep(i,MAX_N) euler[i] = i;
     for(int i = 2; i < MAX_N; i++){
         if(euler[i] == i){
             for(int j = i; j < MAX_N; j += i) euler[j] = euler[j] / i * (i - 1);
         }
     }
 }

snippet     repalceAll
abbr        文字列の置き換え
 string replaceAll(string s, string from, string to){
     vector<int> all;
     string tmp = s, tmp_space = s;
 
     string::size_type pos = tmp.find(from);
     while(pos != string::npos){
         all.emplace_back(pos);
         pos = tmp.find(from, pos + from.size());
     }
 
     //string space(from.size(), ' ');
     rep(j,all.size()){
         tmp.replace(all[j] + (to.size() - from.size()) * j, from.size(), to);
         //tmp_space.replace(all[j] + (from.size() - to.size()) * j, from.size(), space);
     }
     //if(tmp_space.find(to) != string::npos) return "0";
     if(s == tmp || all.empty()) "0";
 
     return tmp;
 }

snippet     JoinInterval
abbr        区間の結合
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }
 
 //Give input directly to vector<pair<int, int>> in
 vector<pair<int, int>> JoinInterval(vector<pair<int,int>> in){
     vector<pair<int, int>> v;
     rep(i,in.size()) in[i].second *= -1;
     sort(all(in));
     rep(i,in.size()) in[i].second *= -1;
 
     rep(i,in.size()){
         if(v.empty()) v.emplace_back(in[i]);
         else{
             pair<int, int> &u = v.back();
             int tmp = intervalState(in[i].first,in[i].second,u.first,u.second);
             switch (tmp){
                 case 0:
                 case 1:
                     v.emplace_back(in[i]);
                     break;
                 case 2:
                     u.first = in[i].first;
                     u.second = in[i].second;
                     break;
                 case 3:
                     break;
                 case 4:
                 case 5:
                     u.first = min(u.first, in[i].first);
                     u.second = max(u.second, in[i].second);
                     break;
                 case -1:
                     assert(0);
             }
         }
     }
     sort(all(v));
     return v;
 }

snippet     closedLoop
abbr        閉路の検出
 const int MAX_V = 505;
 
 vector<int> g[MAX_V];
 vector<int> tp;
 
 bool visit(int v, vector<int> &color){
     color[v] = 1;
     rep(i,g[v].size()){
         int d = g[v][i];
         if(color[d] == 2) continue;
         if(color[d] == 1) return false;
         if(not visit(d, color)) return false;
     }
     tp.emplace_back(v);
     color[v] = 2;
     return true;
 }
 
 bool topologicalSort(int v){
     vector<int> color(v);
     rep(i,v){
         if(not color[i] && not visit(i, color)) return false;
     }
     reverse(all(tp));
     return true;
 }

snippet     treeDP
abbr        全方位木DPによる木の直径の演算
 struct edge {
     int to, cost;
 };
 
 vector< edge > g[100000];
 long long dist[100000];
 
 
 void dfs1(int idx, int par) {
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         dfs1(e.to, idx);
         dist[idx] = max(dist[idx], dist[e.to] + e.cost);
     }
 }
 
 int dfs2(int idx, int d_par, int par) {
     vector< pair< int, int > > d_child;
     d_child.emplace_back(0, -1); // 番兵みたいなアレ
     for(edge &e : g[idx]) {
         if(e.to == par) d_child.emplace_back(d_par + e.cost, e.to);
         else d_child.emplace_back(e.cost + dist[e.to], e.to);
     }
     sort(d_child.rbegin(), d_child.rend());
     int ret = d_child[0].first + d_child[1].first; // 最大から 2 つ
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         // 基本は d_child() の最大が d_par になるが, e.to の部分木が最大値のときはそれを取り除く必要がある
         ret = max(ret, dfs2(e.to, d_child[d_child[0].second == e.to].first, idx));
     }
     return (ret);
 }
 
 int solve(int v/*頂点数*/){
     dfs1(v / 2, - 1);
     return dfs(v / 2, 0, - 1);
 }


snippet     maximumSubarray
abbr        連続した部分列の和の最大値を求める
 int maximumSubarray(vector<int> v){
     int sum = 0, ans = -INF;
     for(auto i : v){
         sum = max(i, sum + i);
         ans = max(ans, sum);
     }
     return ans;
 }

snippet     cumulativeSum2D
abbr        2次元累積和
 template <class T> class CumulativeSum2D {
     public:
         vector<vector<T>> s;
         int h, w;
         CumulativeSum2D(const vector<vector<T>> &a) {
             h = a.size();
             w = a[0].size();
             s = vector<vector<T>>(h + 1, vector<T>(w + 1, 0));
             rep(i, h) rep(j, w) s[i + 1][j + 1] = a[i][j];
             rep(i, h + 1) rep(j, w) s[i][j + 1] += s[i][j];
             rep(i, h) rep(j, w + 1) s[i + 1][j] += s[i][j];
         }
 
         int sum(int i, int j, int h, int w) {
             return s[i + h][j + w] - s[i][j + w] + s[i][j] - s[i + h][j];
         }
 
         int maximumSubMatrix(){
             int ret = -INF;
             rep(i,h + 1) rep(j,w + 1) range(k,1,h - i + 1) range(l,1,w - j + 1){
                 ret = max(ret, sum(i,j,k,l));
             }
             return ret;
         }
 };

snippet     maximumrectangule
abbr        最大長方形
 int maximumRectangule(int h, int w, char m[505][505]){
     int c[505][505] = {{0}};
     rep(i,w){
         int cnt = 0;
         rep(j,h){
             if(m[j][i] == '.'){
                 cnt++;
                 c[j][i] = cnt;
             }else{
                 cnt = 0;
             }
         }
     }
     rep(i,h) c[i][w] = 0;
 
     int ans = 0;
     rep(i,h){
         vector<pair<int, int>> v; //height, pos
         rep(j,w + 1){
             if(v.empty()) v.emplace_back(c[i][j],j);
             else{
                 if(v.back().first < c[i][j]) v.emplace_back(c[i][j],j);
                 else{
                     int target = j;
                     while(not v.empty() && v.back().first >= c[i][j]){
                         ans = max(ans, v.back().first * (j - v.back().second));
                         target = v.back().second;
                         v.pop_back();
                     }
                     v.emplace_back(c[i][j],target);
                 }
             }
         }
     }
     return ans;
 }

snippet     matrixChainMultiplication
abbr        連鎖行列積
 // O(n^3)
 // dp[i][j] := [i,j]を計算するための最小の掛け算の回数
 // i番目の行列の(行数、列数)が、(p[i - 1], p[i])
 int matrixChainMultiplication(int n, int p[105]){
     int dp[105][105] = {{0}};
     range(seg, 2, n + 1){ //計算する行列の個数
         range(i, 1, n - seg + 2){
             int j = i + seg - 1;
             dp[i][j] = INT_MAX;
             range(k,i,j){
                 int cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                 dp[i][j] = min(dp[i][j], cost);
             }
         }
     }
     return dp[1][n];
 }

snippet     nibu-graph
abbr        二部グラフ判定・色降り
options     head
    const int MAX_V = 100005;
    vector<int> G[MAX_V];
    
    int color[MAX_V];
    
    bool dfs(int cur, int c){
    	color[cur] = c;
    	for(auto to : G[cur]){
    		if(color[to] == c) return false;
    		if(color[G[cur][to]] == 0 && not dfs(G[cur][to], -c)) return false;
    	}
    	return true;
    }
    
    void solve(int n){
    	rep(i,n){
    		if(color[i] == 0){
    			if(not dfs(i, 1)){
    				cout << "No" << endl; //二部グラフではない
    				return;
    			}
    		}
    	}
    	cout << "Yes" << endl; //二部グラフ
    }

snippet     bitSubSet
abbr        部分集合のビット列を降順に列挙する
 int sub = s;
 do {
 
 	sub = (sub - 1) & s;
 }while(sub != s);

snippet     bitSizeSet
abbr        0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
 //0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
 int comb = (1 << k) - 1;
 while(comb < (1 << n)){
 	int x = comb & -comb, y = comb + x;
 	comb = ((comb & ~y) / x >> 1) | y;
 }

snippet     AhoCorasick
abbr        文字列を扱う木構造
 const int MAX = 26;
 const char OFFSET = 'a';
 
 struct Node{
 	int nxt[MAX+1];			// 次のalphabeteのノード番号
 	int exist;				// 子ども以下に存在する文字列の数の合計
 	vector<int> accept;		// その文字列id
 	Node() : exist(0){memset(nxt, -1, sizeof(nxt));}
 };
 
 class Trie{
 	private:
 		void update_direct(int node,int id){
 			nodes[node].accept.push_back(id);
 		}
 		void update_child(int node,int child,int id){
 			++nodes[node].exist;
 		}
 		void add(const string &str,int str_index,int node_index,int id){
 			if(str_index == str.size())
 				update_direct(node_index, id);
 			else{
 				const int c = str[str_index] - OFFSET;
 				if(nodes[node_index].nxt[c] == -1) {
 					nodes[node_index].nxt[c] = (int) nodes.size();
 					nodes.push_back(Node());
 				}
 				add(str, str_index + 1, nodes[node_index].nxt[c], id);
 				update_child(node_index, nodes[node_index].nxt[c], id);
 			}
 		}
 
 	public:
 		vector<Node>nodes;
 		int root;
 		Trie() : root(0){nodes.push_back(Node());}
 		void add(const string &str,int id){add(str, 0, 0, id);}
 		void add(const string &str){add(str, nodes[0].exist);}
 		int size(){return (nodes[0].exist);}
 		int nodesize(){return ((int) nodes.size());}
 };
 
 class AhoCorasick : public Trie{
 	public: 
 		static const int FAIL = MAX;
 		vector<int> correct;
 		AhoCorasick() : Trie() {}
 
 		void build(){
 			correct.resize(nodes.size());
 			rep(i,nodes.size())correct[i]=(int)nodes[i].accept.size();
 
 			queue<int> que;
 			rep(i,MAX+1){
 				if(~nodes[0].nxt[i]) {
 					nodes[nodes[0].nxt[i]].nxt[FAIL] = 0;
 					que.emplace(nodes[0].nxt[i]);
 				}else nodes[0].nxt[i] = 0;
 			}
 			while(!que.empty()) {
 				Node now = nodes[que.front()];
 				correct[que.front()] += correct[now.nxt[FAIL]];
 				que.pop();
 				rep(i,MAX){
 					if(now.nxt[i] == -1) continue;
 					int fail = now.nxt[FAIL];
 					while(nodes[fail].nxt[i] == -1) {
 						fail = nodes[fail].nxt[FAIL];
 					}
 					nodes[now.nxt[i]].nxt[FAIL] = nodes[fail].nxt[i];
 
 					auto &u = nodes[now.nxt[i]].accept;
 					auto &v = nodes[nodes[fail].nxt[i]].accept;
 					vector<int> accept;
 					set_union(all(u),all(v),back_inserter(accept));
 					u=accept;
 					que.emplace(now.nxt[i]);
 				}
 			}
 		}
 		int match(const string &str,vector<int> &result,int now=0){
 			result.assign(size(),0);
 			int count=0;
 			for(auto &c:str) {
 				while(nodes[now].nxt[c-OFFSET]==-1)now=nodes[now].nxt[FAIL];
 				now = nodes[now].nxt[c-OFFSET];
 				count += correct[now];
 				for(auto &v:nodes[now].accept)result[v]++;
 			}
 			return count;
 		}
 		int next(int now,char c){
 			while(nodes[now].nxt[c-OFFSET]==-1)now=nodes[now].nxt[FAIL];
 			return nodes[now].nxt[c-OFFSET];
 		}
 };


snippet     eulerGraph
abbr        オイラー路の構築
 class EulerGraph{
 	private:
 		bool directed;
 		bool isList;
 		int degreeCheck(int a){
 			if(a == 0) return EULER_GRAPH;
 			else if(a == 2) return SEMI_EULER_GRAPH;
 			else return 0;
 		}
 		int degreeCheck(vector<vector<int>> d){
 			int s = 0, t = 0, p = 0;
 			for(auto i : d){
 				if(i[0] == i[1] + 1) s++;
 				else if(i[0] + 1 == i[1]) t++;
 				else if(i[0] == i[1]) p++;
 				else return 0;
 			}
 			if(s == 0 && t == 0) return EULER_GRAPH;
 			else if(s == 1 && t == 1) return SEMI_EULER_GRAPH;
 			else return 0;
 		}
 		int checkEulerDirectedList(){
 			vector<vector<int>> degree(g.size(), vector<int>(2,0)); //0:out, 1:in
 			rep(i,g.size()){
 				degree[i][0] = g[i].size();
 				for(auto j : g[i]){
 					degree[j][1]++;
 				}
 			}
 			return degreeCheck(degree);
 		}
 		int checkEulerIndirectedList(){
 			int a = 0;
 			rep(i,g.size()){
 				if(g[i].size() % 2) a++;
 			}
 			return degreeCheck(a);
 		}
 		int checkEulerDirectedMat(){
 			vector<vector<int>> degree(g.size(), vector<int>(2,0)); //0:out, 1:in
 			rep(i,g.size()){
 				rep(j,g[i].size()){
 					if(g[i][j] > 0){
 						degree[i][0]++;
 						degree[j][1]++;
 					}
 				}
 			}
 			return degreeCheck(degree);
 		}
 		int checkEulerIndirectedMat(){
 			int a = 0;
 			rep(i,g.size()){
 				int degree = 0;
 				rep(j,g[i].size()){
 					if(g[i][j] > 0) degree++;
 				}
 				if(degree % 2) a++;
 			}
 			return degreeCheck(a);
 		}
 		//gが隣接リスト
 		//gを破壊する
 		//始点、隣接リスト、有向辺かどうか
 		vector<int> ListEulerianTrail(const int s, const bool directed) {
 			function<void (int, vector<int> &)> dfs = [&](int u, vector<int> &trail) {
 				while (not g[u].empty()) {
 					int v = g[u].back();
 					g[u].pop_back();
 					if (not directed) {
 						for (int i = 0; i < g[v].size(); i ++) {
 							if (g[v][i] == u) {
 								g[v].erase(g[v].begin() + i);
 								break;
 							}
 						}
 					}
 					dfs(v, trail);
 				}
 				trail.emplace_back(u);
 			};
 			vector<int> trail;
 			dfs(s, trail);
 			reverse(trail.begin(), trail.end());
 			return trail;
 		}
 		//gが隣接行列
 		//gを破壊する
 		vector<int> MatEulerianTrail(const int s, const bool directed) {
 			function<void (int, vector<int> &)> dfs = [&](int u, vector<int> &trail) {
 				for (int v = 0; v < g.size(); v ++) if (g[u][v] > 0) {
 					g[u][v]--;
 					if (not directed) g[v][u]--;
 					dfs(v, trail);
 				}
 				trail.push_back(u);
 			};
 			vector<int> trail;
 			dfs(s, trail);
 			reverse(trail.begin(), trail.end());
 			return trail;
 		}
 	public:
 		vector<vector<int>> g;
 		EulerGraph(vector<vector<int>> &g, bool directed, bool isList){
 			this->g = g;
 			this->directed = directed;
 			this->isList = isList;
 		}
 
 		static const int EULER_GRAPH = 1;
 		static const int SEMI_EULER_GRAPH = 2;
 		int isEulerGraph(){
 			if(isList){
 				if(directed) return checkEulerDirectedList();
 				else return checkEulerIndirectedList();
 			}else{
 				if(directed) return checkEulerDirectedMat();
 				else return checkEulerIndirectedMat();
 			}
 			return -1;
 		}
 		vector<int> getEulerianTrail(const int s){
 			if(isList) return ListEulerianTrail(s,directed);
 			else return MatEulerianTrail(s, directed);
 		}
 };

snippet     grundy
abbr        グランディー数を求める再帰関数
 int memo[101];
 int grundy(int n){
 	auto& p = memo[n];
 	if(p != -1) return p;
 
 	set<int> s;
 
 
 	int res = 0;
 	while(s.count(res)) res++;
 	return p = res;
 }

snippet     dijkstra-extend
abbr        拡張dijkstra
 //https://beta.atcoder.jp/contests/code-thanks-festival-2015-open/tasks/code_thanks_festival_2015_g
 
 class Edge{ //頂点、コスト、辺に与えられた識別番
 	public:
 		int to, cost, color;
 		Edge(int to, int cost, int color) : to(to) ,cost(cost) ,color(color) {}
 };
 
 class Node{
 	public:
 		int dis;
 		bool isUsed;
 		Node(){
 			this->dis = INF;
 			this->isUsed = 0;
 		}
 };
 
 class Dijkstra{
 	private:
 		vector<vector<Edge>> g;
 		vector<map<int, Node>> node; // node[i][j] := 頂点iに色jで来たとき
 		int n;
 	public:
 		Dijkstra(int n){
 			this->n = n;
 			g = vector<vector<Edge>>(n);
 			node = vector<map<int, Node>>(n);
 		}
 		int solve(int start, int goal, int spre){	//spre スタート時の状態。何もなければ-1
 			typedef pair<int,pair<int, int>> ppi;
 			priority_queue<ppi, vector<ppi>, greater<ppi>> q;
 
 			q.push(make_pair(0, make_pair(start, spre)));
 			node[start][spre].dis = 0;
 
 			ppi u;
 			while(not q.empty()){
 				u = q.top(); q.pop();
 				int cur = u.second.first;
 				int pre = u.second.second;
 				node[cur][pre].isUsed = 1;
 				if(cur == goal) break;//これがないとTLE
 				//breakしないとgoalに到着したコストを各色について求めてしまう
 				//最後の色の指定がないので、どんな色でもいいから最短経路が決まったらbreakする。
 
 				for(const auto& e : g[cur]){
 					int to = e.to;
 					int color = e.color;
 					int cost = e.cost + abs(pre - color) + u.first;
 
 					auto& next = node[to][color];
 					if(next.isUsed == 0){
 						if(not node[to].count(color) || next.dis > cost){
 							next.dis = cost;
 							q.push(make_pair(next.dis, make_pair(to, color)));
 						}
 					}
 				}
 			}
 
 			int mini = INF;
 			for(auto it : node[goal]){
 				mini = min(mini, it.second.dis);
 			}
 			return mini;
 		}
 		void addEdge(int from, int to, int cost, int color){
 			g[from].emplace_back(Edge{to, cost, color});
 			g[to].emplace_back(Edge{from, cost, color});
 		}
 };

snippet     rightHand
abbr        右手法
 // 右手法
 // right[d] := d方向（東から時計回り）を向いているときの優先順序
 const int rightHand[4][4] = {
 	{1,0,3,2},
 	{2,1,0,3},
 	{3,2,1,0},
 	{0,3,2,1}
 };
