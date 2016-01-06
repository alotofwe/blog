title: 競プロ用C++ライブラリ-幾何以外
date: 2015-12-05 14:01:00
tags:
categories: 
- 競プロ
- library
---

ほとんど[Spaghetti Source](http://www.prefield.com/algorithm/)のコピペです．

```
/////////////////////////////
// 構文解析

P fact(char *p){
  if(isdigit(*p)){
    int t = *(p++) - '0';
    while(isdigit(*p)) t = t * 10 + *(p++) - '0';
    return P(t, p);
  } else if(*p == '('){
    P r = expr(p + 1);
    if(*r.second != ')') exit(0); //閉じ括弧が無いエラー
    return P(r.first, r.second + 1);
  } else{
    exit(0); //括弧でも数字でもないエラー
  }
}

P term(char *p){
  P r = fact(p);
  while(*r.second == '*' || *r.second == '/'){
    char op = *r.second;
    int tmp = r.first;
    r = fact(r.second + 1);
    if(op == '*') r.first *= tmp;
    else r.first /= tmp;
  }
  return r;
}

P expr(char *p){
  P r = term(p);
  while(*r.second == '+' || *r.second == '-'){
    char op = *r.second;
    int tmp = r.first;
    r = term(r.second + 1);
    if(op == '+') r.first = tmp + r.first;
    else r.first = tmp - r.first;
  }
  return r;
}

//////////////////////////////
// 最大公約数

int gcd(int a, int b){
  if(b == 0) return a;
  else return gcd(b, a % b);
}

//////////////////////////////
// 最小公倍数(gcdが必要)

int lcm(int a, int b){
  return a / gcd(a, b) * b;
}

//////////////////////////////
// 文字列分割

vector<string> split(string &str, char delim){
  stringstream ss(str);
  string tmp;
  vector<string> res;
  while(getline(ss, tmp, delim)) res.push_back(tmp);
  return res;
}

//////////////////////////////
// 繰り返し二乗法(xのn乗)

Int mod_pow(Int x, Int n, Int mod){
  if(n == 0) return 1;
  Int ret = mod_pow(x * x % mod, n / 2, mod);
  if(n & 1) ret = ret * x % mod;
  return ret;
}

//////////////////////////////
// 行列累乗

typedef vector<int> vec;
typedef vector<vec> mat;

long long n;
//H = 行列の縦, W = 行列の横, M = modをとるときに使う(無い場合は消す)
int H, W, M = 1;

//A*B
mat mul(mat &A, mat &B){
  mat C(A.size(), vec(B[0].size()));
  REP(i, A.size())
    REP(k, B.size())
      REP(j, B[0].size())
        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % M;
  return C;
}

//A^n
mat pow(mat A, long long n){
  mat B(A.size(), vec(A.size()));
  REP(i, A.size())
    B[i][i] = 1;
  while(n > 0){
    if(n & 1) B = mul(B, A);
    A = mul(A, A);
    n >>= 1;
  }
  return B;
}

void solve(){
  cin >>H >>W;
  mat A(H, vec(W));
  //ここで定式化した行列を代入する
  A = pow(A, n);
}

//////////////////////////////
// UnionFind

vector<int> v, parent, depth;

int find(int a){
  if(parent[a] == a) return a;
  else return parent[a] = find(parent[a]);
}

void merge(int a, int b){
  int pa = find(a), pb = find(b);
  if(pa == pb) return ;

  if(depth[pa] > depth[pb]) swap(pa, pb);
  if(depth[pa] == depth[pb]) ++depth[pa];
  parent[pb] = pa;
}

bool same(int a, int b){
  return find(a) == find(b);
}

void init_union_find(int V){
  v = vector<int>(V);
  parent = vector<int>(V);
  depth = vector<int>(V, 1);
  REP(i, V) parent[i] = i;
}

//////////////////////////////
// UnionFind (重み付き)

map<int, int> v, parent, dist;
map<string, int> M;

// P(parent, dist)
P find(int a){
  if(parent[a] == a) return P(a, dist[a]);
  P p = find(parent[a]);
  parent[a] = p.first;
  dist[a] += p.second;
  return P(p.first, dist[a]);
}

bool same(int a, int b){ return find(a).first == find(b).first; }

void merge(int a, int b, int dd){
  if(same(a, b)) return ;
  P pa = find(a), pb = find(b);
  int d = dd - pb.second + pa.second;
  if(d < 0){
    d *= -1;
    swap(pa, pb);
  }
  parent[pb.first] = pa.first;
  dist[pb.first] = d;
}

int distance(int a, int b){
  if(!same(a, b)) return INF;
  return find(b).second - find(a).second;
}

void init_union_find(){
  v = map<int, int>();
  parent = map<int, int>();
  dist = map<int, int>();
  M = map<string, int>();
}

//////////////////////////////
// ベルマン-フォード

struct edge { int from, to, cost; };
edge es[MAX_E];
int d[MAX_V], V, E;

//負閉路が存在する場合にfalseを返す
bool bellman_ford(int s){
  REP(i, V) d[i] = INF;
  d[s] = 0;
  REP(i, V + 1){
    bool update = false;
    REP(j, E){
      edge e = es[j];
      if(d[e.from] != INF && d[e.to] > d[e.from] + e.cost){
        d[e.to] = d[e.from] + e.cost;
        if(i == V) return false;
        update = true;
      }
    }
  }
  return true;
}

//////////////////////////////
// ワーシャル-フロイド法

int V;

//負閉路が存在する場合にfalseを返す
bool warshall_floyd(long c[MAX_V][MAX_V]){
  REP(k, V)
    REP(i, V)
      REP(j, V)
        if(c[i][k] != INF && c[k][j] != INF) c[i][j] = min(c[i][j], c[i][k] + c[k][j]);
  REP(i, V) if(c[i][i] < 0) return false;
  return true;
}

//////////////////////////////
// クラスカル法(Unionfindあるのが前提)

struct edge {
  int f, t, c;
  bool operator < (const edge &e) const { return c < e.c; };
};

int kruskal(int V, int E, vector<edge> es){
  sort(es.begin(), es.end());
  init_union_find(V);
  int min_cost = 0;
  REP(i, E){
    if(!same(es[i].f, es[i].t)){
      min_cost += es[i].c;
      merge(es[i].f, es[i].t);
    }
  }
  return min_cost;
}

//////////////////////////////
// プリム法

int prim(int v[N][N]){
  priority_queue< P, vector<P>, greater<P> > open;
  open.push(P(0, 0));
  vector<int> closed(V, INF);
  int min_cost = 0;
  while(!open.empty()){
    P now = open.top(); open.pop();
    if(closed[now.second] != INF) continue;
    closed[now.second] = now.first;
    min_cost += now.first;
    REP(i, V){
      if(closed[i] == INF && v[now.second][i] != INF) open.push(P(v[now.second][i], i));
    }
  }
  return min_cost;
}

//////////////////////////////
// セグメント木

const int MAX_N = (1 << 17);
const int INF = 2147483647;

// 要素の数、区間などを何からの値を覚えておく配列
int n, v[2 * MAX_N - 1];

// 初期化
void init(int _n){
  n = 1;
  while(n < _n) n *= 2;
  REP(i, 2 * n - 1) v[i] = INF;
}

// 添字がkのものをaに更新
void update(int k, int a){
  k += n - 1;
  v[k] = a;
  while(k > 0){
    k = (k - 1) / 2;
    v[k] = min(v[k * 2 + 1], v[k * 2 + 2]);
  }
}

// [a, b)の最小値を求める
// kは今見ている添字、l, rはそれに対応する[l, r)
// 呼び出す時はquery(a, b, 0, 0, n)として呼ぶ
int query(int a, int b, int k, int l, int r){
  if(b <= l || a >= r) return INF;
  if(a <= l && b >= r) return v[k];
  int vl = query(a, b, k * 2 + 1, l, (l + r) / 2);
  int vr = query(a, b, k * 2 + 2, (l + r) / 2, r);
  return min(vl, vr);
}


//////////////////////////////
// Binary Indexed Tree (数列の添字は1はじまり)

int bit[MAX_N + 1], n;

void init(){
  memset(bit, 0, sizeof(bit));
}

int sum(int i){
  int ret = 0;
  while(i > 0){
    ret += bit[i];
    i -= (i & -i);
  }
  return ret;
}

void add(int i, int x){
  while(i <= n){
    bit[i] += x;
    i += (i & -i);
  }
}

//////////////////////////////
// フロー(Dinic)
//

const int MAX_V = 305;
const int INF = 1e9 + 7;

class Edge{
  public:
    int to, cap, rev;
    Edge(int _to, int _cap, int _rev){ to = _to; cap = _cap; rev = _rev; }
};

vector<Edge> G[MAX_V];
int level[MAX_V], iter[MAX_V];

void add_edge(int from, int to, int cap){
  G[from].push_back(Edge(to, cap, G[to].size()));
  G[to].push_back(Edge(from, 0, G[from].size() - 1));
}

void bfs(int s){
  memset(level, -1, sizeof(level));
  queue<int> q;
  level[s] = 0;
  q.push(s);
  while(!q.empty()){
    int v = q.front(); q.pop();
    REP(i, G[v].size()){
      Edge &e = G[v][i];
      if(e.cap > 0 && level[e.to] < 0){
        level[e.to] = level[v] + 1;
        q.push(e.to);
} } } }

int dfs(int v, int t, int f){
  if(v == t) return f;
  for(int &i = iter[v]; i < G[v].size(); ++i){
    Edge &e = G[v][i];
    if(e.cap > 0 && level[v] < level[e.to]){
      int d = dfs(e.to, t, min(f, e.cap));
      if(d > 0){
        e.cap -= d;
        G[e.to][e.rev].cap += d;
        return d;
  } } }
  return 0;
}

int max_flow(int s, int t){
  int flow = 0;
  while(true){
    bfs(s);
    if (level[t] < 0) return flow;
    memset(iter, 0, sizeof(iter));
    int f;
    while((f = dfs(s, t, INF)) > 0)
      flow += f;
  }
}

//////////////////////////////
// kD Tree (nth_elementが遅く、ノード数が10^5とかだと約9秒かかる)

struct kdTree {
  int removed;
  struct Node {
    int x, y, n;
    bool removed;
    Node *ch[2];
    int size, ht;
    Node(int x, int y, int n) : x(x), y(y), n(n), removed(false) { ch[0] = ch[1] = 0; }
    Node() : x(-1), y(-1), n(-1), removed(false) { ch[0] = ch[1] = 0; }
  } *r;
  int size(Node *t) { return t ? t->size : 0; }
  int ht(Node *t) { return t ? t->ht : 0; }
  Node *update(Node *t) {
    if (t == NULL) return t;
    t->size = 1 + size(t->ch[0]) + size(t->ch[1]);
    t->ht = 1 + max(ht(t->ch[0]), ht(t->ch[1]));
    return t;
  }
  struct Compare {
    int d;
    Compare(int d) : d(d) { }
    bool operator()(Node *i, Node *j) {
      return d == 0 ? (i->x <= j->x) : (i->y <= j->y);
    }
  };
  Node *build(Node **b, Node **e, int d) {
    if (e - b <= 0) return 0;
    //qsort(b, (size_t)(e - b), sizeof((*b)), d % 2 ? cmp_y : cmp_x);
    Node **m = b + (e - b) / 2;
    nth_element(b, m, e, Compare(d)); 
    //for(Node **i = b; i != e; ++i) cout <<(**i).n <<", "; cout <<endl;
    //cout <<(**m).n <<endl;
    (*m)->ch[0] = build(b, m, (d+1)%D);
    (*m)->ch[1] = build(m+1, e, (d+1)%D);
    return update(*m);
  }
  Node **flatten(Node *r, Node **buf) {
    if (!r) return buf;
    buf = flatten(r->ch[0], buf);
    if (!r->removed) *(buf++) = r;
    return flatten(r->ch[1], buf);
  }
  Node *rebuild(Node *r, int d) {
    Node *b[size(r)], **e = flatten(r, b);
    removed -= size(r) - (e - b);
    return build(b, e, d);
  }
  Node *insert(Node *t, Node *p, int d) {
    if (!t) return update(p);
    int b = !Compare(d)(p, t); 
    t->ch[b] = insert(t->ch[b], p, (d+1)%D);
    t = update(t);
    if (3 * log(size(t)) < ht(t)) t = rebuild(t, d);
    return t;
  }
  void insert(int x, int y) { r = insert(r, new Node(x,y,size(r)), 0); }
  Node *find(Node *t, Node *p, int d) {
    if (!t) return 0;
    Node *f = 0;
    if (!t->removed && t->x == p->x && t->y == p->y) f = t;
    if (!f && !Compare(d)(p,t)) f = find(t->ch[1], p, (d+1)%D);
    if (!f && !Compare(d)(t,p)) f = find(t->ch[0], p, (d+1)%D);
    return f;
  }
  Node *find(int x, int y) { Node n(x,y,-1); return find(r, &n, 0); }
  void count(Node *t, Node *S, Node *T, int d, int *cnt, int idx[MAX_N]){
    if (!t) return ;
    if (!t->removed && t->x >= S->x && t->x <= T->x && t->y >= S->y && t->y <= T->y) idx[(*cnt)++] = t->n;
    if (Compare(d)(S,t)) count(t->ch[0], S, T, (d+1)%D, cnt, idx);
    if (Compare(d)(t,T)) count(t->ch[1], S, T, (d+1)%D, cnt, idx);
  }
  void remove(int x, int y) { 
    Node *f = find(x, y);
    if (!f) return;
    f->removed = true;
    ++removed;
    if (removed*2 > r->size) r = rebuild(r, 0);
  }
  kdTree(int p[][2], int n) : removed(0) {
    Node *node[n];
    for (int i = 0; i < n; ++i) 
      node[i] = new Node(p[i][0], p[i][1], i);
    r = build(node, node+n, 0);
  }
};

//////////////////////////////
// ダイクストラ

int dijkstra(const vector< vector<P> > &cost, int s, int t, int V){
  priority_queue<P, vector<P>, greater<P> > open;
  open.push(P(0, s));
  int closed[MAX_V];
  REP(i, V) closed[i] = INF;
  while(!open.empty()){
    P tmp = open.top(); open.pop();
    int now = tmp.second, c = tmp.first;
    if(closed[now] < c) continue;
    closed[now] = c;
    REP(i, cost[now].size()){
      int next = cost[now][i].second, nc = cost[now][i].first;
      if(nc == INF || c + nc >= closed[next]) continue;
      closed[next] = c + nc;
      open.push(P(closed[next], next));
    }
  }
  return closed[t];
}


//////////////////////////////
// 関節点列挙 (関節点とは、その点を削除するとグラフが2つに分かれてしまうような点

void dfs(int u, int prev, int *cnt, int *visited, int *prenum, int *lowest, int *parent, vector< vector<int> > &v){
  visited[u] = true;
  prenum[u] = lowest[u] = *cnt;
  REP(i, v[u].size()){
    int next = v[u][i];
    if(!visited[next]){
      parent[next] = u;
      ++(*cnt);
      dfs(next, u, cnt, visited, prenum, lowest, parent, v);
      lowest[u] = min(lowest[u], lowest[next]);
    } else if(next != prev)
      lowest[u] = min(lowest[u], prenum[next]);
  }
}

vector<int> articulation_points(int V, int E, vector< vector<int> > &v){
  int visited[V], prenum[V], lowest[V], parent[V];
  memset(visited, 0, sizeof(visited));
  parent[0] = -1;
  int tmp = 1;
  dfs(0, -1, &tmp, visited, prenum, lowest, parent, v);
  vector<int> ret;
  int cnt = 0, used[V];
  memset(used, 0, sizeof(used));
  FOR(i, 1, V){
    if(parent[i] == 0) ++cnt;
    if(parent[i] > 0 && !used[parent[i]] && prenum[parent[i]] <= lowest[i]){
      used[parent[i]] = true;
      ret.push_back(parent[i]);
    }
  }
  if(cnt >= 2) ret.push_back(0);
  return ret;
}

//////////////////////////////
// 橋列挙 (橋とは、その辺を削除するとグラフが2つに分かれてしまうような辺

int dfs(int u, int prev, int *cnt, int *prenum, int *lowest, vector< vector<int> > &v, vector<P> &res){
  prenum[u] = lowest[u] = ++(*cnt);
  REP(i, v[u].size()){
    int next = v[u][i];
    if(prenum[next] == -1){
      lowest[u] = min(lowest[u], dfs(next, u, cnt, prenum, lowest, v, res));
      if(lowest[next] == prenum[next]) res.push_back(P(min(u, next), max(u, next)));
    }
    else if(prev != next)
      lowest[u] = min(lowest[u], lowest[next]);
  }
  return lowest[u];
}

vector<P> bridges(int V, int E, vector< vector<int> > &v){
  int prenum[V], lowest[V];
  memset(prenum, -1, sizeof(prenum));
  int tmp = 1;
  vector<P> res;
  dfs(0, -1, &tmp, prenum, lowest, v, res);
  return res;
}

//////////////////////////////
// 強連結成分分解

void dfs(int v, int *used, vector< vector<int> > &G, vector<int> &vs){
  used[v] = true;
  REP(i, G[v].size()){
    int next = G[v][i];
    if(!used[next]) dfs(next, used, G, vs);
  }
  vs.push_back(v);
}

void rdfs(int v, int cnt, int *used, vector<int> &cmp, vector< vector<int> > &G){
  used[v] = true;
  cmp[v] = cnt;
  REP(i, G[v].size()){
    int next = G[v][i];
    if(!used[next]) rdfs(next, cnt, used, cmp, G);
  }
}

//Gは元々の有向グラフの隣接リスト、RGは辺の向きを逆にした隣接リスト
//返り値は、任意のノードが何番目の強連結成分に属するか。cmp[a] == cmp[b]を満たせばa, bは同じ成分に属する。
vector<int> scc(int V, vector< vector<int> > &G, vector< vector<int> > &RG){
  int used[V];
  memset(used, 0, sizeof(used));
  vector<int> vs, cmp(V);
  REP(i, V) if(!used[i]) dfs(i, used, G, vs);
  memset(used, 0, sizeof(used));
  int cnt = 0;
  REVERSE(vs);
  REP(i, vs.size()) if(!used[vs[i]]) rdfs(vs[i], cnt++, used, cmp, RG);
  return cmp;
}

//////////////////////////////
// 閉路検出

//コストを全て-1にして各ノードについてベルマンフォード (コード略)

//////////////////////////////
// 木の直径

P visit(int v, int prev, vector< vector<P> > &G) {
  P res(v, 0);
  REP(i, G[v].size()){
    if(G[v][i].first != prev){
      P t = visit(G[v][i].first, v, G);
      t.second += G[v][i].second;
      if (res.second < t.second) res = t;
    }
  }
  return res;
}
int diameter(vector< vector<P> > &G) {
  P r = visit(0, -1, G);
  P t = visit(r.first, -1, G);
  return t.second; // (r.first, t.first) is farthest pair
}

//////////////////////////////
// 最小有向全域木

void visit(int V, int G[MAX_V][MAX_V], int v, int s, int r,
    vector<int> &no, vector< vector<int> > &comp,
    vector<int> &prev, vector< vector<int> > &next,
    vector<int> &mcost, vector<int> &mark,
    int &cost, bool &found) {
  if (mark[v]) {
    vector<int> temp = no;
    found = true;
    do {
      cost += mcost[v];
      v = prev[v];
      if (v != s) {
        while (comp[v].size() > 0) {
          no[comp[v].back()] = s;
          comp[s].push_back(comp[v].back());
          comp[v].pop_back();
        }
      }
    } while (v != s);
    REP(i, V){
      if(i != r && no[i] == s)
        REP(j, V){
          if (no[j] != s && G[j][i] < INF) G[j][i] -= mcost[temp[i]];
        }
    }
  }
  mark[v] = true;
  REP(i, next[v].size())
    if(no[next[v][i]] != no[v] && prev[no[next[v][i]]] == v)
      if (!mark[no[next[v][i]]] || next[v][i] == s)
        visit(V, G, next[v][i], s, r, no, comp, prev, next, mcost, mark, cost, found);

}

int minimumSpanningArborescence(int V, int E, int G[MAX_V][MAX_V], int r) {
  vector<int> no(V);
  vector< vector<int> > comp(V);
  REP(i, V) comp[i].push_back(no[i] = i);

  for(int cost = 0; ;) {
    vector<int> prev(V, -1);
    vector<int> mcost(V, INF);
    REP(i, V){
      REP(j, V){
        if(j == r || G[i][j] == INF || no[i] == no[j] || G[i][j] > mcost[no[j]]) continue;
        mcost[no[j]] = G[i][j];
        prev[no[j]] = no[i];
      }
    }
    vector< vector<int> > next(V);
    REP(i, V)
      if(prev[i] >= 0)
        next[prev[i]].push_back(i);
    bool stop = true;
    vector<int> mark(V, false);
    REP(i, V)
      if(i != r && !mark[i] && !comp[i].empty()) {
        bool found = false;
        visit(V, G, i, i, r, no, comp, prev, next, mcost, mark, cost, found);
        if (found) stop = false;
      }
    if (stop) {
      REP(i, V) if (prev[i] >= 0) cost += mcost[i];
      return cost;
    }
  }
}

int main() {
  int V, E, r;
  cin >>V >>E >>r;
  int G[MAX_V][MAX_V];
  REP(i, V) REP(j, V) G[i][j] = (i == j ? 0 : INF);
  REP(i, E){
    int f, t, c; cin >>f >>t >>c;
    G[f][t] = c;
  }
  cout <<minimumSpanningArborescence(V, E, G, r) <<endl;
  return 0;
}

//////////////////////////////
// 2部マッチング(容量が全て1)

int V, match[MAX_V], used[MAX_V];
vector<int> G[MAX_V];

bool dfs(int v){
  used[v] = true;
  REP(i, G[v].size()){
    int u = G[v][i], w = match[u];
    if(w < 0 || (!used[w] && dfs(w))){
      match[v] = u;
      match[u] = v;
      return true;
    }
  }
  return false;
}

int bipartite_matching(){
  int res = 0;
  memset(match, -1, sizeof(match));
  REP(v, V)
    if(match[v] < 0){
      memset(used, 0, sizeof(used));
      if(dfs(v)) ++res;
    }
  return res;
}

int main() {
  int X, Y, E; cin >>X >>Y >>E;
  V = X + Y;
  REP(i, E){
    int u, v; cin >>u >>v;
    v += X;
    G[u].push_back(v);
    G[v].push_back(u);
  }
  cout <<bipartite_matching() <<endl;
  return 0;
}

//////////////////////////////
// コインDP
// (複数の金額について出力せよという問題が多いと思うので、答えは返さずDP表をつくるだけ)

//d = コインの種類の数
void coinDP(int d, int *coins, int *dp){
  REP(i, MAX_COST + 1) dp[i] = INF;
  dp[0] = 0;
  REP(i, d)
    FOR(j, coins[i], MAX_COST+ 1)
      dp[j] = min(dp[j], dp[j - coins[i]] + 1);
}


//////////////////////////////
// ナップザック(入れるものは1種類につき1つ)

//N = 入れるものの数, W = 許される最大の重量
int knapsack(int N, int W, int *v, int *w){
  int dp[N + 1][W + 1];
  memset(dp, -1, sizeof(dp));
  dp[0][0] = 0;
  REP(i, N){
    REP(j, W + 1){
      if(dp[i][j] < 0) continue;
      dp[i + 1][j] = max(dp[i + 1][j], dp[i][j]);
      if(j + w[i] <= W) dp[i + 1][j + w[i]] = max(dp[i + 1][j + w[i]], dp[i][j] + v[i]);
    }
  }
  int ans = 0;
  REP(i, W + 1) ans = max(ans, dp[N][i]);
  return ans;
}

//////////////////////////////
// ナップザック(入れるものは1種類につきいくつでも)

int multiple_knapsack(int N, int W, int *v, int *w){
  int dp[N + 1][W + 1];
  memset(dp, -1, sizeof(dp));
  REP(i, W + 1) dp[0][i] = 0;
  REP(i, N){
    REP(j, W + 1){
      if(dp[i][j] < 0) continue;
      dp[i + 1][j] = max(dp[i + 1][j], dp[i][j]);
      if(j - w[i] >= 0) dp[i + 1][j] = max(dp[i + 1][j], dp[i + 1][j - w[i]] + v[i]);
      if(j + w[i] <= W) dp[i + 1][j + w[i]] = max(dp[i + 1][j + w[i]], dp[i][j] + v[i]);
    }
  }
  int ans = 0;
  REP(i, W + 1) ans = max(ans, dp[N][i]);
  return ans;
}

//////////////////////////////
// 最長増加部分列

int LIS(int N, int *a){
  int dp[N];
  fill(dp, dp + N, INF);
  REP(i, N) *lower_bound(dp, dp + N, a[i]) = a[i];
  return lower_bound(dp, dp + N, INF) - dp;
}

//////////////////////////////
// 編集距離(レーベンシュタイン距離)

int LD(string &A, string &B){
  int al = A.length(), bl = B.length(), dp[al + 1][bl + 1];
  REP(i, al + 1) REP(j, bl + 1) dp[i][j] = (i == 0 || j == 0 ? max(i, j) : INF);
  FOR(i, 1, al + 1)
    FOR(j, 1, bl + 1)
      dp[i][j] = min(dp[i][j - 1] + 1, min(dp[i - 1][j] + 1, dp[i - 1][j - 1] + (A[i - 1] == B[j - 1] ? 0 : 1)));
  return dp[al][bl];
}

//////////////////////////////
// 巡回セールスマン (巡回できない場合は-1を返す)

int traveling_salesman(int V, int G[MAX_V][MAX_V]){
  int dp[(1 << V)][V];
  REP(i, (1 << V)) REP(j, V) dp[i][j] = INF;
  dp[0][0] = 0;
  REP(i, (1 << V)){
    REP(j, V){
      REP(k, V){
        int mask = (1 << k), cost = G[j][k];
        if((i & mask) || cost < 0) continue;
        dp[(i | mask)][k] = min(dp[(i | mask)][k], dp[i][j] + cost);
      }
    }
  }
  return (dp[(1 << V) - 1][0] == INF ? -1 : dp[(1 << V) - 1][0]);
}

//////////////////////////////
// 無向中国人郵便配達問題 (グラフの全ての辺を少なくとも一度通る単純とは限らない閉路の中で，最短のものを求める)

void dijkstra(const vector< vector<P> > &cost, int V, int s, int *closed){
  priority_queue<P, vector<P>, greater<P> > open;
  open.push(P(0, s));
  REP(i, V) closed[i] = INF;
  closed[s] = 0;
  while(!open.empty()){
    P tmp = open.top(); open.pop();
    int now = tmp.second, c = tmp.first;
    if(closed[now] < c) continue;
    closed[now] = c;
    REP(i, cost[now].size()){
      int next = cost[now][i].second, nc = cost[now][i].first;
      if(nc == INF || c + nc >= closed[next]) continue;
      closed[next] = c + nc;
      open.push(P(closed[next], next));
    }
  }
}

int chinesePostman(const int &V, const vector< vector<P> > &G) {
  int total = 0;
  vector<int> odds;
  REP(i, V) {
    REP(j, G[i].size())
      total += G[i][j].first;
    if ((int)(G[i].size()) % 2) odds.push_back(i);
  }
  total /= 2;
  int n = odds.size(), N = (1 << n);
  int w[n][n]; // make odd vertices graph
  REP(i, n) {
    int closed[V];
    dijkstra(G, V, odds[i], closed);
    REP(j, n) w[i][j] = closed[odds[j]];
  }
  int dp[N]; // DP for general matching 
  REP(S, N) dp[S] = INF;
  dp[0] = 0;
  for (int S = 0; S < N; ++S)
    REP(i, n)
      if (!(S & (1 << i)))
        FOR(j, i + 1, n)
          if (!(S & (1 << j)))
            dp[(S | (1 << i) | (1 << j))] = min(dp[(S | (1 << i) | (1 << j))], dp[S] + w[i][j]);
  return total + dp[N - 1];
}

//////////////////////////////
// 素因数分解

//昇順に並んで返される
vector<int> prime_factorize(int N){
  vector<int> ret;
  FOR(i, 2, sqrt(N) + 1){
    while(!(N % i)){
      ret.push_back(i);
      N /= i;
    }
  }
  if(N > 1) ret.push_back(N);
  return ret;
}

//////////////////////////////
// オイラーのφ関数 (正の整数nについて、1 からnまでの自然数のうちnと互いに素なものの個数)

int eulers_phi(int N){
  int ret = N;
  for(int i = 2; i * i <= N; ++i){
    if(!(N % i)){
      ret -= ret / i;
      while(!(N % i)) N /= i;
    }
  }
  if(N > 1) ret -= ret / N;
  return ret;
}

//////////////////////////////
// 拡張ユークリッドの互除法 (ax + by = 1となる整数x, yを求める)
// 蟻本第1版p120参照

int extgcd(int a, int b, int &x, int &y){
  int d = a;
  if(b != 0){
    d = extgcd(b, a % b, y, x);
    y -= (a / b) * x;
  } else{
    x = 1;
    y = 0;
  }
  return d;
}

//////////////////////////////
// にぶたん

int binary_search(int X){
  int l = -1, r = N;
  while(r - l > 1){
    int Y = (l + r) / 2;
    if(/*条件*/) r = Y;
    else l = Y;
  }
  return r;
}

//////////////////////////////
// LCA (木の2頂点の共通祖先)

const int MAX_V = 100010;
const int MAX_LOG_V = 17;

vector<int> G[MAX_V];
int root = 0;

int parent[MAX_LOG_V][MAX_V];
int depth[MAX_V];

void dfs(int v, int p, int d){
  parent[0][v] = p;
  depth[v] = d;
  for(int i = 0; i < G[v].size(); ++i) if (G[v][i] != p) dfs(G[v][i], v, d + 1);
}

void init(int V){
  dfs(root, -1, 0);
  for(int k = 0; k + 1 < MAX_LOG_V; ++k){
    for(int v = 0; v < V; ++v){
      if(parent[k][v] < 0) parent[k + 1][v] = -1;
      else parent[k + 1][v] = parent[k][parent[k][v]];
    }
  }
}

int lca(int u, int v){
  if(depth[u] > depth[v]) swap(u, v);
  for(int k = 0; k < MAX_LOG_V; ++k){
    if((depth[v] - depth[u]) >> k & 1) v = parent[k][v];
  }
  if(u == v) return u;
  for(int k = MAX_LOG_V - 1; k >= 0; --k){
    if(parent[k][u] != parent[k][v]){
      u = parent[k][u];
      v = parent[k][v];
    }
  }
  return parent[0][u];
}

//////////////////////////////
// 木の2頂点間距離 (LCAがあること前提)

// rootからの距離
int dist[MAX_V];

void make_dist(int n, int now, int d){
  dist[now] = d;
  REP(i, G[now].size()){
    int next = G[now][i];
    if(dist[next] != -1) continue;
    make_dist(n, next, d + 1);
  }
}

int solve(int u, int v){
  memset(dist, -1, sizeof(dist));
  int N; cin >>N;
  // 木の隣接リストの入力
  REP(i, N - 1){
    int a, b; cin >>a >>b;
    G[a].push_back(b);
    G[b].push_back(a);
  }
  make_dist(N, 0, 0);
  init(N);
  return (dist[u] + dist[v] - 2 * dist[lca(u, v)]);
}

//////////////////////////////
// 座標圧縮

long long int v[MAX_YX][MAX_YX];

int main(){
  memset(v, 0, sizeof(v));
  vector<int> L, T, R, B;
  set<int> Ys, Xs;
  REP(i, N){
    int l, t, r, b;
    cin >>l >>t >>r >>b;
    L.push_back(l);
    T.push_back(t);
    R.push_back(r);
    B.push_back(b);
    Ys.insert(t);
    Ys.insert(b);
    Xs.insert(l);
    Xs.insert(r);
  }
  Ys.insert(-1);
  Ys.insert(1e6 + 10);
  Xs.insert(-1);
  Xs.insert(1e6 + 10);
  vector<int> Y(Ys.begin(), Ys.end());
  vector<int> X(Xs.begin(), Xs.end());
  REP(i, N){
    int xl = lower_bound(X.begin(), X.end(), L[i]) - X.begin();
    int xr = lower_bound(X.begin(), X.end(), R[i]) - X.begin();
    int yt = lower_bound(Y.begin(), Y.end(), T[i]) - Y.begin();
    int yb = lower_bound(Y.begin(), Y.end(), B[i]) - Y.begin();
    FOR(x, xl, xr){
      FOR(y, yb, yt){
        long long int bit = (1LL << i);
        v[x][y] |= bit;
      }
    }
  }
}

//////////////////////////////
// 二次元累積和

void init(){
  REP(i, H) REP(j, W) cin >>v[i][j];
  memset(E, 0, sizeof(E));
  E[0][0] = v[0][0];
  FOR(i, 1, H) E[i][0] = E[i - 1][0] + v[i][0];
  FOR(i, 1, W) E[0][i] = E[0][i - 1] + v[0][i];
  FOR(y, 1, H)
    FOR(x, 1, W)
    E[y][x] = v[y][x] + E[y - 1][x] + E[y][x - 1] - E[y - 1][x - 1];
}

int calc(int Y1, int X1, int Y2, int X2){
  int ret = E[Y2][X2];
  if(X1 - 1 >= 0) ret -= E[Y2][X1 - 1];
  if(Y1 - 1 >= 0) ret -= E[Y1 - 1][X2];
  if(X1 - 1 >= 0 && Y1 - 1 >= 0) ret += E[Y1 - 1][X1 - 1];
  return ret;
}

//////////////////////////////
// iCjの組み合わせ

double C[MAX_N][MAX_N];

void pascals_triangle(int N){
  REP(i, N + 1) C[i][0] = C[i][i] = 1.0;
  FOR(i, 1, N + 1){
    FOR(j, 1, i + 1)
      C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
  }
}
//////////////////////////////
// iCjの確率

double C[MAX_N][MAX_N];

void pascals_triangle(int N){
  C[0][0] = 1.0;
  FOR(i, 1, N + 1){
    C[i][0] = C[i - 1][0] / 2.0;
    FOR(j, 1, i + 1)
      C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) / 2.0;
  }
}

//////////////////////////////
// B^N

double power(double B, int N){
  double ret = B;
  REP(i, N) ret *= B;
  return ret;
}

//////////////////////////////
// Nの倍数

ref: http://www004.upp.so-net.ne.jp/s_honma/number/multiple.htm

２	・・・	　一の位が２の倍数
３	・・・	　各位の数の和が３の倍数
４	・・・	　・下二桁が４の倍数
              ・一の位を２で割った数を十の位に足した数が偶数
５	・・・	　一の位が５の倍数
６	・・・	　２かつ３の倍数
７	・・・	　・３桁毎に交互に足したり引いたりしてできた数が７の倍数
              ・３桁の数 ａｂｃ で、ａｂ－２ｃ が７の倍数（→詳細は、こちら）
              　一般に、１０ｐ＋ｑ において、ｐ－２ｑ が７の倍数（→詳細はこちら）
              ・３桁の数 ａｂｃ で、２ａ＋ｂｃ が７の倍数（→詳細は、こちら）
              　一般に、１００ｐ＋ｑ において、２ｐ＋ｑ が７の倍数（→詳細はこちら）
              ・６桁の場合、２桁毎に７で割った余りを考え、それらの数で出来る２桁の
              　整数の差が７の倍数（→詳細は、こちら）
８	・・・	　・下３桁が８の倍数
              ・一の位を２で割り十の位に足して２で割った数を百の位に足した数が偶数
９	・・・	　各位の数の和が９の倍数
１０	・・・	　一の位が０
１１	・・・	　各位の数を交互に足したり引いたりしてできた数が１１の倍数(奇数桁目の合計と偶数桁目の合計の差が11の倍数)
１２	・・・	　３かつ４の倍数
１３	・・・	　７の倍数の判定と同じ
１４	・・・	　２かつ７の倍数
１５	・・・	　３かつ５の倍数
１６	・・・	　下４桁を２で割った数が８の倍数（下４桁を４で割った数が４の倍数）
１７	・・・	　・十位以上の数から一位の数の５倍を引いた数が１７の倍数
                ・２桁毎に下位から２のべきを掛けて交互に足したり引いたりしてできた数が１７の倍数
１８	・・・	　２かつ９の倍数
１９	・・・	　各位の数に上位から２のべきを掛けて足した数が１９の倍数
２０	・・・	　４かつ５の倍数
２１	・・・	　３かつ７の倍数
２２	・・・	　２かつ１１の倍数
２３	・・・	　十位以上の数と一位の数の７倍の和が２３の倍数
２４	・・・	　３かつ８の倍数
３７	・・・	　３桁毎に区分けした数を足した数が３７の倍数
９９９	・・・	３桁毎に区分けした数を足した数が９９９の倍数


//////////////////////////////
// サイコロ
 
int dx[4] = {0,1,-1,0};
int dy[4] = {-1,0,0,1};

int H[6][6] = {
    {1,5,2,3,0,4}, // North : 奥へ移動   ( y:-1 )
    {3,1,0,5,4,2}, // East  : 右へ移動   ( x:+1 )
    {2,1,5,0,4,3}, // West  : 左へ移動   ( x:-1 )
    {4,0,2,3,5,1}, // South : 手前へ移動 ( y:+1 )
    {0,2,4,1,3,5}, // Right : 右回りに回転 (移動なし)
    {0,3,1,4,2,5}  // Left  : 左回りに回転 (移動なし)
};
 
// サイコロライブラリ
// d[0] := top,   d[1] := front 
// d[2] := right, d[3] := left
// d[4] := back,  d[5] := bottom
enum{TOP, FRONT, RIGHT, LEFT, BACK, BOTTOM};
struct Cube{
    vector<int> d;
    // コンストラクタで初期化
    Cube(vector<int> v){
        if( v.size() == 6 )
            d = v;
        else
            d = vector<int>(6);
    }
    Cube(){ d = vector<int>(6); }
    // dirの方向に回転 (副作用なし)
    Cube roll(int dir){
        vector<int> d_(6);
        for(int i = 0 ; i < 6 ; i++ ){
            d_[i] = d[ H[dir][i] ];
        }
        return Cube(d_);
    }
};
// Cube の順序を定義 (map<Cube,T> を使いたいとき用)
bool operator<(const Cube& a, const Cube& b){
    if( a.d[0] == b.d[0] )
        return a.d[1] < b.d[1];
    return a.d[0] < b.d[0];
}

//////////////////////////////
// 最小費用流

typedef pair<int, int> P;

struct edge { int to, cap, cost, rev; }

int V;
vector<edge> G[MAX_V]; //グラフの隣接リスト表現
int h[MAX_V]; //ポテンシャル
int dist[MAX_V]; //最短距離
int prevv[MAX_V], preve[MAX_V]; // 直前の頂点と辺

void add_edge(int from, int to, int cap, int cost){
  G[from].push_back((edge){to, cap, cost, G[to].size())});
  G[to].push_back((edge){from, 0, -cost, (int)(G[from].size()) - 1});
}

// sからtへの流量fの最小費用流
// 流せない場合は-1を返す
int min_cost_flow(int s, int t, int f){
  int res = 0;
  fill(h, h + V, 0);
  while(f > 0){
    priority_queue<P, vector<P>, greater<P> > que;
    fill(dist, dist + V, INF);
    dist[s] = 0;
    que.push(P(0, s));
    while(!que.empty()){
      P p = que.top(); que.pop();
      int v = p.second;
      if(dist[v] < p.first) continue;
      REP(i, G[v].size()){
        edge &e = G[v][i];
        if(e.cap > 0 && dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]){
          dist[e.to] = dist[v] + e.cost + h[v] - h[e.to];
          prevv[e.to] = v;
          preve[e.to] = i;
          que.push(P(dist[e.to], e.to));
        }
      }
    }
    if(dist[t] == INF) return -1;
    REP(v, V) h[v] += dist[v];
    int d = f;
    for(int v = t; v != s; v = prevv[v]) d = min(d, G[prevv[v]][preve[v]].cap);
    f -= d;
    res += d * h[t];
    for(int v = t; v != s; v = prevv[v]){
      edge &e = G[prevv[v]][preve[v]];
      e.cap -= d;
      G[v][e.rev].cap += d;
    }
  }
  return res;
}


    }

}

//////////////////////////////
// ローリングハッシュ

typedef unsigned long long ull;

const ull B = 100000007;

// aはbに含まれているか
bool contain(string a, string b){
  int al = a.length(), bl = b.length();
  if(al > bl) return false;

  ull t = 1;
  REP(i, al) t *= B;

  ull ah = 0, bh = 0;
  REP(i, al) ah = ah * B + a[i];
  REP(i, al) bh = bh * B + b[i];

  for(int i = 0; i + al <= bl; ++i){
    if(ah == bh) return true; // bのi9文字目からのal文字が一致
    if(i + al < bl) bh = bh * B + b[i + al] - b[i] * t;
  }
  return false;
}

// aの末尾とbの先頭を何文字重ねることができるか
int overlap(string a, string b){
  int al = a.length(), bl = b.length();
  int ans = 0;
  ull ah = 0, bh = 0, t = 1;
  for(int i = 1; i <= min(al, bl); ++i){
    ah = ah + a[al - i] * t;
    bh = bh * B + b[i - 1];
    if(ah == bh) ans = i;
    t *= B;
  }
  return ans;
}

//////////////////////////////
// Trie木 (rootはmainの先頭で*root = new nodeとして宣言)

typedef struct node {
  node *ch[3];
  bool exist;
  node() { memset(ch, 0, sizeof(ch)); exist = false; }
} node;

void add(node *root, string &s){
  REP(i, s.length()){
    int n = s[i] - 'a';
    if(root->ch[n] == NULL) root->ch[n] = new node;
    root = root->ch[n];
  }
  root->exist = true;
}

bool solve(node *root, string &s, int pos, int cnt){
  if(pos >= s.length()) return (root->exist && cnt == 1);
  int n = s[pos] - 'a';
  REP(i, 3){
    if(cnt < 1 && root->ch[i]) if(solve(root->ch[i], s, pos + 1, (n == i ? cnt : cnt + 1))) return true;
    if(cnt >= 1 && n == i && root->ch[i]) if(solve(root->ch[i], s, pos + 1, cnt)) return true;
  }
  return false;
}

//////////////////////////////
// 有理数

typedef long long Integer;
Integer gcd(Integer a, Integer b) { return a > 0 ? gcd(b % a, a) : b; }
struct rational {
  Integer p, q;
  void normalize() { // keep q positive
    if (q < 0) p *= -1, q *= -1;
    Integer d = gcd(p < 0 ? -p : p, q);
    if (d == 0) p = 0,  q = 1;
    else        p /= d, q /= d;
  }
  rational(Integer p, Integer q = 1) : p(p), q(q) {
    normalize();
  }
  rational &operator += (const rational &a) {
    p = a.q * p + a.p * q; q = a.q * q; normalize();
    return *this;
  }
  rational &operator -= (const rational &a) {
    p = a.q * p - a.p * q; q = a.q * q; normalize();
    return *this;
  }
  rational &operator *= (const rational &a) {
    p *= a.p; q *= a.q; normalize();
    return *this;
  }
  rational &operator /= (const rational &a) {
    p *= a.q; q *= a.p; normalize();
    return *this;
  }
  rational &operator - () {
    p *= -1;
    return *this;
  }
};
rational operator + (const rational &a, const rational &b) {
  return rational(a) += b;
}
rational operator * (const rational &a, const rational &b) {
  return rational(a) *= b;
}
rational operator - (const rational &a, const rational &b) {
  return rational(a) -= b;
}
rational operator / (const rational &a, const rational &b) {
  return rational(a) /= b;
}
bool operator < (const rational &a, const rational &b) { // avoid overflow
  return (long double) a.p * b.q < (long double) a.q * b.p;
}
bool operator <= (const rational &a, const rational &b) {
  return !(b < a);
}
bool operator > (const rational &a, const rational &b) {
  return b < a;
}
bool operator >= (const rational &a, const rational &b) {
  return !(a < b);
}
bool operator == (const rational &a, const rational &b) {
  return !(a < b) && !(b < a);
}
bool operator != (const rational &a, const rational &b) {
  return (a < b) || (b < a);
}
```
