#diameterOfTree
# 木の直径
#unionFind-WeightedVer
# 頂点重み付き
#unionFind
# 集合
#minCostFlow
# 最小費用流
#lca
# lowest common ancestor
#twoEdgeConnectedComponent
# 二重連結成分分解、無向辺の強連結成分分解
#articulationPoins
# 関節点と橋を全列挙
#dijkstra-list
# 隣接リストのダイクストラ
#kruskal
# クラスカル法による最小全域木構築
#dijkstra-matrix
# 隣接行列のダイクストラ
#stronglyConnectedComponents
# 強連結成分分解

snippet     stronglyConnectedComponents
abbr        強連結成分分解
 class StronglyConnectedComponents{
 	private:
 		vector<bool> used;
 		vector<int> vs; //帰りがけ順の並び
 		void dfs(int v){
 			used[v] = true;
 			rep(i,g[v].size()){
 				if(not used[g[v][i]]) dfs(g[v][i]);
 			}
 			vs.emplace_back(v);
 		}
 		void rdfs(int v, int k){
 			used[v] = true;
 			cmp[v] = k;
 			rep(i,rg[v].size()){
 				if(not used[rg[v][i]]) rdfs(rg[v][i], k);
 			}
 		}
 	public:
 		typedef vector<vector<int>> graph;
 		const int v; // 頂点数
 		int nv; // SCCした後の頂点数
 		graph g, rg; // グラフ、辺が逆になったグラフ
 		vector<int> cmp; //属する強連結成分のトポロジカル順序
 
 		StronglyConnectedComponents(int v) : used(v), v(v), g(v), rg(v), cmp(v) { }
 
 		void addEdge(int from, int to){
 			g[from].emplace_back(to);
 			rg[to].emplace_back(from);
 		}
 		int solve(){ // 強連結成分分解をしたあとのグラフの頂点数を返す
 			fill(all(used),0);
 			vs.clear();
 			rep(i,v){
 				if(not used[i]) dfs(i);
 			}
 			fill(all(used),0);
 			int k = 0;
 			for(int i = vs.size() - 1; i >= 0; i--){
 				if(not used[vs[i]]) rdfs(vs[i], k++);
 			}
 			return nv = k;
 		}
 		graph getCssGraph(vector<vector<int>>& node){
 			node = vector<vector<int>>(nv); // node[i]:=SCCによって頂点iに同一視された頂点
 			graph res(nv); // CSSしたあとのグラフ
 			rep(i,v){
 				node[cmp[i]].emplace_back(i);
 				for(auto to : g[i]){
 					if(cmp[i] == cmp[to]) continue;
 					res[cmp[i]].emplace_back(cmp[to]);
 				}
 			}
 			return res;
 		}
 		void out(){
 			rep(i,v){ cout << cmp[i] << ' '; } cout << endl;
 		}
 };

snippet     dijkstra-matrix
abbr        隣接行列のダイクストラ
 const int MAX_V = 1000;
 int dijkstra(int n, int s, int g[MAX_V][MAX_V]){
 	vector<int> dis(n,INF);
 	priority_queue<int, vector<pair<int, int>>, greater<pair<int, int>>> q;
 
 	q.emplace(0,s);
 	dis[s] = 0;
 
 	while(not q.empty()){
 		int pos;
 		int cost;
 		tie(cost, pos) = q.top(); q.pop();
 		if(pos == 1) break;
 
 		rep(to,n){
 			if(dis[to] > cost + g[pos][to]){
 				dis[to] = cost + g[pos][to];
 				q.emplace(dis[to], to);
 			}
 		}
 	}
 	return dis[1];
 }

snippet     kruskal
abbr        クラスカル法による最小全域木構築
 class UnionFind{
 	private:
 		vector<int> par, depth;
 	public:
 		UnionFind() {}
 		UnionFind(int n){
 			init(n);
 		}
 		void init(int n){
 			par = vector<int>(n);
 			depth = vector<int>(n);
 			rep(i,n){
 				par[i] = i;
 				depth[i] = 0;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 			}else{
 				par[y] = x;
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 };
 
 struct Edge{
 	int u, v, cost, id;
 	Edge() {}
 	Edge(int u, int v, int cost) : u(u), v(v), cost(cost) {}
 
 	bool operator < ( const Edge &right ) const {
 		return cost < right.cost ? 1 : (cost == right.cost ? (u < right.u ? 1 : (u == right.u ? (v == right.v) : 0)) : 0);
 	}
 };
 
 class Kruskal{
 	private:
 		static const int MAX_E = 50005;
 		UnionFind unionFind;
 	public:
 		int v;
 		vector<Edge> edge;
 		vector<int> mst; //edge[ mst[i] ] := 最小全域木を構築するi番目の辺
 		Kruskal(vector<Edge>& e, int _v){
 			v = _v;
 			unionFind.init(v);
 			edge = e;
 			sort(all(edge));
 		}
 		void reset(int v){
 			unionFind.init(v);
 		}
 		long long solve(){
 			long long res = 0;
 			rep(i,edge.size()){
 				Edge c = edge[i];
 				if(not unionFind.same(c.u, c.v)){
 					unionFind.unite(c.u, c.v);
 					res += c.cost;
 					mst.emplace_back(i);
 				}
 			}
 			int b = 0;
 			rep(i,v){
 				if(sinked[i]) continue;
 				b = i;
 				break;
 			}
 			rep(i,v){
 				if(not unionFind.same(b,i)){
 					return INF;
 				}
 			}
 			return res;
 		}
 };

snippet     dijkstra-list
abbr        隣接リストのダイクストラ
 struct Edge{
 	int to;
 	int cost;
 	Edge(int to, int cost) : to(to), cost(cost) {}
 };
 
 struct Node{
 	int dis;
 	bool used;
 	Node() : dis(INF), used(false) { }
 	Node(int d, bool f) : dis(d), used(f) { }
 };
 
 typedef vector<vector<Edge>> graph;
 
 int dijkstra(graph g, int s, int n){
 	vector<Node> node(n);
 	priority_queue<int, vector<pair<int, int>>, greater<pair<int, int>>> q;
 
 	q.push(make_pair(0, s));
 	node[s] = Node{0, true};
 
 	while(not q.empty()){
 		int dis, pos;
 		tie(dis, pos) = q.top(); q.pop();
 		node[pos].used = true;
 
 		for(auto e : g[pos]){
 			if(node[e.to].used == true) continue;
 			if(node[e.to].dis > dis + e.cost){
 				node[e.to].dis = dis + e.cost;
 				q.emplace(node[e.to].dis, e.to);
 			}
 		}
 	}
 	return node[0].dis;
 }


snippet     articulationPoins
abbr        関節点と橋を全列挙
 class Node{
 	public:
 		int ord; //DFSの訪問の順番
 		int par; //DFS Treeにおける親
 		int low; //min(自分のord, 逆辺がある場合の親のord, すべての子のlow)
 		Node() : ord(-1) { }
 };
 
 class ArticulationPoints {
 	private:
 		int v, cnt;
 		vector<Node> node;
 		void dfs(int cur, int prev){
 			node[cur].ord = node[cur].low = cnt;
 			cnt++;
 
 			for(auto to : g[cur]){
 				if(to == prev) continue;
 				if(node[to].ord >= 0){
 					node[cur].low = min(node[cur].low, node[to].ord);
 				}else{
 					node[to].par = cur;
 					dfs(to, cur);
 					node[cur].low = min(node[cur].low, node[to].low);
 				}
 				if(node[cur].ord < node[to].low){
 					bridge.emplace_back(min(cur, to), max(cur, to));
 				}
 			}
 		}
 	public:
 		vector<vector<int>> g;
 		vector<pair<int, int>> bridge;
 		set<int> ap; // 関節店
 		ArticulationPoints(int n) : v(n), cnt(1), node(n), g(n) { }
 		void addEdge(int a, int b){
 			g[a].emplace_back(b);
 			g[b].emplace_back(a);
 		}
 		bool isBridge(int u, int v){
 			if(node[u].ord > node[v].ord) swap(u,v);
 			return node[u].ord < node[v].low;
 		}
 		void run(){
 			dfs(0, -1); // 0 = root
 
 			int np = 0;
 			range(i,1,v){
 				int p = node[i].par;
 				if(p == 0) np++;
 				else if(node[p].ord <= node[i].low) ap.emplace(p);
 			}
 			if(np > 1) ap.emplace(0);
 			sort(all(bridge));
 			//for(auto it:ap){ cout << it << endl; } //関節点の全列挙
 			//for(auto it:bridge){ cout << it.first << ' ' << it.second << endl; } //橋の全列挙
 		}
 };

snippet     twoEdgeConnectedComponent
abbr        二重連結成分分解、無向辺の強連結成分分解
 class TwoEdgeConnectedComponent : public ArticulationPoints {
 	private:
 		void dfs(int c, int pos){
 			ver[c].emplace_back(pos);
 			comp[pos] = c;
 			for (int to : g[pos]) {
 				if (comp[to] >= 0) continue;
 				if (isBridge(pos, to)) continue;
 				dfs(c, to);
 			}
 		}
 		void addComp(int pos){
 			if(comp[pos] >= 0) return;
 			ver.emplace_back();
 			dfs(ver.size() - 1, pos);
 		}
 	public:
 		vector<int> comp; // 成分分解前の頂点から分解後の頂点への写像？
 		vector<vector<int>> ver; // 分解後の頂点と、その頂点に含まれる分解前の頂点
 		TwoEdgeConnectedComponent(int n) : ArticulationPoints(n), comp(n,-1) {}
 		void solve(){
 			run();
 			for(auto p : bridge){
 				addComp(p.first);
 				addComp(p.second);
 			}
 			addComp(0);
 		}
 		vector<vector<int>> getTree(){
 			vector<vector<int>> res(ver.size());
 			for(auto p : bridge){
 				int u = comp[p.first], v = comp[p.second];
 				res[u].emplace_back(v);
 				res[v].emplace_back(u);
 			}
 			return res;
 		}
 };

snippet     lca
abbr        lowest common ancestor
 class LCA{
 	private:
 		vector<vector<int>> g, parent;
 		int n, root, log_n;
 		bool inited;
 		vector<int> depth;
 		void dfs(int v, int p, int d){
 			parent[0][v] = p;
 			depth[v] = d;
 			rep(i,g[v].size()){
 				if(g[v][i] != p) dfs(g[v][i], v, d + 1);
 			}
 		}
 		void init(){
 			inited = true;
 			dfs(root, -1, 0);
 			rep(k,log_n - 1){
 				rep(i,n){
 					if(parent[k][i] < 0) parent[k + 1][i] = -1;
 					else parent[k + 1][i] = parent[k][ parent[k][i] ];
 				}
 			}
 		}
 		void dfs(int pos){
 			for(auto to : g[pos]){
 				if(dis[to] != -1) continue;
 				dis[to] = dis[pos] + 1;
 				dfs(to);
 			}
 		}
 	public:
 		vector<int> dis;
 		LCA(int n, int root = 0) : g(n), parent(log2(n) + 1, vector<int>(n)), n(n), root(root), log_n(log2(n) + 1), inited(false), depth(n), dis(n, -1) { }
 		void addEdge(int u, int v){
 			g[u].emplace_back(v);
 			g[v].emplace_back(u);
 		}
 		void dfs(){
 			dis[0] = 0;
 			dfs(0);
 		}
 		int dist(int u, int v){
 			return dis[u] + dis[v] - 2 * dis[get(u,v)];
 		}
 		int get(int u, int v){
 			if(not inited) init();
 			if(depth[u] > depth[v]) swap(u, v);
 			rep(k,log_n){
 				if( (depth[v] - depth[u]) >> k & 1){
 					v = parent[k][v];
 				}
 			}
 			if(u == v) return u;
 			for(int k = log_n - 1; k >= 0; k--){
 				if(parent[k][u] != parent[k][v]){
 					u = parent[k][u];
 					v = parent[k][v];
 				}
 			}
 			return parent[0][u];
 		}
 };

snippet     minCostFlow
abbr        最小費用流
 const int INF = 1e9;
 
 class Edge{
 	public:
 		//行き先、容量、コスト、逆辺
 		int to, cap, cost, rev;
 		Edge(int to, int cap, int cost, int rev) : to(to), cap(cap), cost(cost), rev(rev){}
 };
 
 class MinCostFlow {
 	int v;
 	vector<vector<Edge>> G;
 	vector<int> h; //ポテンシャル
 	vector<int> dist; //最短距離
 	vector<int> prev_v, prev_e; //直前の頂点と辺
 
 	public:
 	MinCostFlow(int n = 500) : v(n), G(n), h(n), dist(n), prev_v(n), prev_e(n) { }
 	void addEdge(int from, int to, int cap, int cost){
 		//cout << from << ' ' << to << ' ' << cap << ' ' << cost << endl;
 		G[from].emplace_back(Edge(to, cap, cost, static_cast<int>(G[to].size())));
 		G[to].emplace_back(Edge(from, 0, -cost, static_cast<int>(G[from].size() - 1)));
 	}
 	int bellmanFord(int s, int t, int f){
 		int res = 0;
 		while(f > 0){
 			dist = vector<int>(v,INF);
 			dist[s] = 0;
 			bool update = true;
 			while(update){
 				update = false;
 				rep(i,v){
 					if(dist[i] == INF) continue;
 					rep(j,G[i].size()){
 						Edge &e = G[i][j];
 						if(e.cap > 0 and dist[e.to] > dist[i] + e.cost){
 							dist[e.to] = dist[i] + e.cost;
 							prev_v[e.to] = i;
 							prev_e[e.to] = j;
 							update = true;
 						}
 					}
 				}
 			}
 			if(dist[t] == INF){
 				return -1;
 			}
 
 			int d = f;
 			for(int u = t; u != s; u = prev_v[u]){
 				d = min(d, G[prev_v[u]][prev_e[u]].cap);
 			}
 			f -= d;
 			res += d * dist[t];
 			for(int u = t; u != s; u = prev_v[u]){
 				Edge &e = G[prev_v[u]][prev_e[u]];
 				e.cap -= d;
 				G[u][e.rev].cap += d;
 			}
 		}
 		return res;
 	}
 	int dijkstra(int s, int t, int f){
 		int res = 0;
 		h = vector<int>(v, 0);
 		while(f > 0){
 			priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
 			dist = vector<int>(v,INF);
 			dist[s] = 0;
 			q.push(make_pair(0, s));
 			while(not q.empty()){
 				pair<int, int> p = q.top(); q.pop();
 				int u = p.second;
 				if(dist[u] < p.first) continue;
 				rep(i,G[u].size()){
 					Edge &e = G[u][i];
 					if(e.cap > 0 && dist[e.to] > dist[u] + e.cost + h[u] - h[e.to]){
 						dist[e.to] = dist[u] + e.cost + h[u] - h[e.to];
 						prev_v[e.to] = u;
 						prev_e[e.to] = i;
 						q.push(make_pair(dist[e.to], e.to));
 					}
 				}
 			}
 			if(dist[t] == INF){
 				return -1;
 			}
 			rep(i,v) h[i] += dist[i];
 
 			int d = f;
 			for(int u = t; u != s; u = prev_v[u]){
 				d = min(d, G[prev_v[u]][prev_e[u]].cap);
 			}
 			f -= d;
 			res += d * h[t];
 			for(int u = t; u != s; u = prev_v[u]){
 				Edge &e = G[prev_v[u]][prev_e[u]];
 				e.cap -= d;
 				G[u][e.rev].cap += d;
 			}
 		}
 		return res;
 	}
 };

snippet     unionFind
abbr        集合
 class UnionFind{
 	private:
 		vector<int> par, depth;
 	public:
 		vector<int> cnt; // その集合の頂点数
 		UnionFind() {}
 		UnionFind(int n){
 			init(n);
 		}
 		void init(int n){
 			par = vector<int>(n);
 			depth = vector<int>(n, 0);
 			cnt = vector<int>(n, 1);
 			rep(i,n){
 				par[i] = i;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 				cnt[y] += cnt[x];
 			}else{
 				par[y] = x;
 				cnt[x] += cnt[y];
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 };

snippet     unionFind-WeightedVer
abbr        頂点重み付き
 template<typename T>
 class UnionFind{
 	private:
 		vector<int> par, depth;
 		vector<T> sum; // 集合の重みの総和
 	public:
 		vector<int> cnt; // その集合の頂点数
 		WeightedUnionFind() {}
 		WeightedUnionFind(int n, vector<T>& c){
 			init(n, c);
 		}
 		void init(int n, vector<T>& c){
 			par = vector<int>(n);
 			depth = vector<int>(n, 0);
 			cnt = vector<int>(n, 1);
 			sum = c;
 			rep(i,n){
 				par[i] = i;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 				cnt[y] += cnt[x];
 				sum[y] += sum[x];
 			}else{
 				par[y] = x;
 				cnt[x] += cnt[y];
 				sum[x] += sum[y];
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 		T weight(int x){
 			return sum[find(x)];
 		}
 };

snippet     diameterOfTree
abbr        木の直径
 const int INF = 1e8;
 
 class Tree{
 	int n;
 	void bfs(int s, vector<int>& dis, vector<int>& pre){
 		queue<int> q;
 		dis[s] = 0;
 		q.emplace(s);
 
 		while(not q.empty()){
 			int u = q.front(); q.pop();
 			show(u)
 			for(auto e : g[u]){
 				if(dis[e.first] == INF){
 					dis[e.first] = dis[u] + e.second;
 					pre[e.first] = u;
 					q.emplace(e.first);
 				}
 			}
 		}
 	}
 	public:
 	vector<vector<pair<int, int>>> g;
 	vector<int> path;
 	Tree(int n) : n(n), g(n) {}
 	void addEdge(int a, int b, int c = 1){
 		g[a].emplace_back(b, c);
 		g[b].emplace_back(a, c);
 	}
 	int diameter(){
 		vector<int> pre(n, -1), dis(n, INF);
 		bfs(0, dis, pre);
 
 		int maxi = 0, tgt = 0;
 		rep(i,n){
 			if(dis[i] == INF) continue;
 			if(maxi < dis[i]){
 				maxi  = dis[i];
 				tgt = i;
 			}
 		}
 
 		pre = vector<int>(n, -1);
 		dis = vector<int>(n, INF);
 		bfs(tgt, dis, pre);
 		maxi = 0; //tgtから最も遠い接点の距離maxi
 		rep(i,n){
 			if(dis[i] == INF) continue;
 			if(maxi < dis[i]){
 				maxi  = dis[i];
 				tgt = i;
 			}
 		}
 
 		int pos = tgt;
 		while(pre[pos] != -1){
 			path.emplace_back(pre[pos]);
 			pos = pre[pos];
 		}
 		return maxi;
 	}
 	pair<int, int> center(){
 		if(path.empty()) diameter();
 
 		assert(path.size() == 0 and "path is empty");
 		if(path.size() % 2 == 0){
 			return make_pair(path[ path.size() / 2 - 1], path[ path.size() / 2]);
 		}else{
 			return make_pair(path[ path.size() / 2], -1);
 		}
 	}
 };
