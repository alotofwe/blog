title: 競プロ用C++ライブラリ-幾何
date: 2015-12-05 14:01:24
tags:
categories: 
- 競プロ
- library
---

ほとんど[Spaghetti Source](http://www.prefield.com/algorithm/)のコピペです．

```
const double EPS = 1e-8;
const double INF = 1e12;
#define X real()
#define Y imag()
#define curr(P, i) P[i]
#define next(P, i) P[(i+1)%P.size()]
#define prev(P, i) P[(i+P.size()-1) % P.size()]
#define diff(P, i) (next(P, i) - curr(P, i))
#define EQ(a,b) (abs((a)-(b)) < EPS)
namespace std {
  bool operator < (const P& a, const P& b) {
    return real(a) != real(b) ? real(a) < real(b) : imag(a) < imag(b);
  }
}

//overflow注意
int distance2(P &a, P &b){ return (a.real() - b.real()) * (a.real() - b.real()) * (a.imag() - b.imag()) * (a.imag() - b.imag()); }

//外積
double cross(const P& a, const P& b) { return imag(conj(a)*b); }

//内積
double dot(const P& a, const P& b) { return real(conj(a)*b); }

//線 (線分)
struct L : public vector<P> {
  L(const P &a, const P &b) {
    push_back(a); push_back(b);
  }
};

int ccw(P a, P b, P c) {
  b -= a; c -= a;
  if (cross(b, c) > 0)   return +1;       // counter clockwise
  if (cross(b, c) < 0)   return -1;       // clockwise
  if (dot(b, c) < 0)     return +2;       // c--a--b on line
  if (norm(b) < norm(c)) return -2;       // a--b--c on line
  return 0;
}

//多角形
typedef vector<P> G;

//円
struct C {
  P p; double r;
  C(const P &p, double r) : p(p), r(r) { }
};


//交差判定達 (L = 直線, P = 点, S = 線分)

bool intersectLL(const L &l, const L &m) {
  return abs(cross(l[1]-l[0], m[1]-m[0])) > EPS || // non-parallel
    abs(cross(l[1]-l[0], m[0]-l[0])) < EPS;   // same line
}

bool intersectLS(const L &l, const L &s) {
  return cross(l[1]-l[0], s[0]-l[0])*       // s[0] is left of l
    cross(l[1]-l[0], s[1]-l[0]) < EPS; // s[1] is right of l
}

bool intersectLP(const L &l, const P &p) {
  return abs(cross(l[1]-p, l[0]-p)) < EPS;
}

bool intersectSS(const L &s, const L &t) {
  return ccw(s[0],s[1],t[0])*ccw(s[0],s[1],t[1]) <= 0 &&
    ccw(t[0],t[1],s[0])*ccw(t[0],t[1],s[1]) <= 0;
}

bool intersectSP(const L &s, const P &p) {
  return abs(s[0]-p)+abs(s[1]-p)-abs(s[1]-s[0]) < EPS; // triangle inequality
}

//-1 交差しない
//0 接する
//1 一箇所で交差する
//2 二箇所で交差する
int intersectSC(const L &s, const C &c) {
  int ins = 0;
  for(int i = 0; i < 2; i++ ) {
    if (abs(s[i]-c.p)<c.r) ins++;
    else if (EQ(abs(s[i]-c.p), c.r)) return 0;
  }
  if (ins == 2) return -1;
  if (ins == 1) return 1;
  double d = distanceLP(s, c.p);
  if (d-c.r > EPS) return -1;
  P nor=(s[0]-s[1]) * P(0, 1);
  if (ccw(c.p, c.p+nor, s[0]) * ccw(c.p, c.p+nor, s[1]) < 0) return 2;
  return -1;
}

//線分lに対する、点pの射影
//射影とは、pからlに垂線を引いた時の交点
P projection(const L &l, const P &p) {
  double t = dot(p-l[0], l[0]-l[1]) / norm(l[0]-l[1]);
  return l[0] + t*(l[0]-l[1]);
}

//線分lに対する、点pの反射
//反射とは、lを対称軸とした時にpと線対称の位置にある点
P reflection(const L &l, const P &p) {
  return p + P(2, 0) * (projection(l, p) - p);
}

//距離達 (L = 直線, P = 点, S = 線分)

double distancePP(const P &a, const P &b){
  return abs(a - b);
}

double distanceLP(const L &l, const P &p) {
  return abs(p - projection(l, p));
}

double distanceLL(const L &l, const L &m) {
  return intersectLL(l, m) ? 0 : distanceLP(l, m[0]);
}

double distanceLS(const L &l, const L &s) {
  if (intersectLS(l, s)) return 0;
  return min(distanceLP(l, s[0]), distanceLP(l, s[1]));
}

double distanceSP(const L &s, const P &p) {
  const P r = projection(s, p);
  if (intersectSP(s, r)) return abs(r - p);
  return min(abs(s[0] - p), abs(s[1] - p));
}

double distanceSS(const L &s, const L &t) {
  if (intersectSS(s, t)) return 0;
  return min(min(distanceSP(s, t[0]), distanceSP(s, t[1])),
      min(distanceSP(t, s[0]), distanceSP(t, s[1])));
}

//交点達
P crosspointLL(const L &l, const L &m) {
  double A = cross(l[1] - l[0], m[1] - m[0]);
  double B = cross(l[1] - l[0], l[1] - m[0]);
  if (abs(A) < EPS && abs(B) < EPS) return m[0]; // same line
  if (abs(A) < EPS) assert(false); // !!!PRECONDITION NOT SATISFIED!!!
  return m[0] + B / A * (m[1] - m[0]);
}

vector<P> crosspointCL(const C &c, const L &l){
  vector<P> res;
  double d = distanceLP(l, c.p);
  if(d < c.r + EPS){
    double len = (d > c.r) ? 0.0 : sqrt(c.r * c.r - d * d);
    P nor = (l[0] - l[1]) / abs(l[0] - l[1]);
    res.push_back(projection(l, c.p) + len * nor);
    res.push_back(projection(l, c.p) - len * nor);
  }
  return res;
}

vector<P> crosspointCS(const C &c, const L &s){
  vector<P> v = crosspointCL(c, l), res;
  REP(k, v.size()) if(ccw(l[0], v[k], l[1]) == -2) res.push_back(v[k]);
  return res;
}

vector<P> crosspointCC(const C &c1, const C &c2){
  vector<P> res;
  double d = abs(c1.p - c2.p);
  double rc = (d * d + c1.r * c1.r - c2.r * c2.r) / (2 * d);
  double dfr = c1.r * c1.r - rc * rc;
  if(EQ(dfr, 0.0)) dfr = 0.0;
  else if(dfr < 0.0) return res;
  double rs = sqrt(dfr);

  P diff = (c2.p - c1.p) / d;
  res.push_back(c1.p + diff * P(rc, rs));
  res.push_back(c1.p + diff * P(rc, -rs));
  return res;
}

//直交かどうか
bool orthogonalLL(const L &l1, const L &l2){
  return (dot(l1[0]-l1[1], l2[0]-l2[1]) <= EPS && dot(l1[0]-l1[1], l2[0]-l2[1]) >= -EPS);
}

//平行かどうか
bool parallel(const L &l1, const L &l2){
  return (cross(l1[0]-l1[1], l2[0]-l2[1]) <= EPS && cross(l1[0]-l1[1], l2[0]-l2[1]) >= -EPS);
}

//多角形の面積
double area2(const G &P) {
  double A = 0.0;
  for (int i = 0; i < P.size(); ++i)
    A += cross(curr(P, i), next(P, i));
  return A / 2.0;
}

//凸性判定
//ただし、凸性の定義(全ての角が180度未満)より、0度の角があるようなものも凸とする。
//convex_hullして点の数を比較したほうが安全な気がする
bool isconvex(const G &P) {
  for (int i = 0; i < P.size(); ++i)
    if (ccw(prev(P, i), curr(P, i), next(P, i)) == 1) return false;
  return true;
}

//点-多角形包含判定
enum { OUT, ON, IN };
int contains(const G &g, const P &p) {
  bool in = false;
  for (int i = 0; i < g.size(); ++i) {
    P a = curr(g,i) - p, b = next(g,i) - p;
    if (imag(a) > imag(b)) swap(a, b);
    if (imag(a) <= 0 && 0 < imag(b))
      if (cross(a, b) < 0) in = !in;
    if (cross(a, b) == 0 && dot(a, b) <= 0) return ON;
  }
  return in ? IN : OUT;
}

//凸包
G convex_hull(G &ps) {
  int n = ps.size(), k = 0;
  sort(ps.begin(), ps.end());
  G ch(2*n);
  for (int i = 0; i < n; ch[k++] = ps[i++]) // lower-hull
    while (k >= 2 && ccw(ch[k-2], ch[k-1], ps[i]) == -1) --k;
  for (int i = n-2, t = k+1; i >= 0; ch[k++] = ps[i--]) // upper-hull
    while (k >= t && ccw(ch[k-2], ch[k-1], ps[i]) == -1) --k;
  ch.resize(k-1);
  return ch;
}


//凸多角形の直径
//直径とは、ある頂点と別の頂点が結ぶ距離の最長のこと
double convex_diameter(const G &pt) {
  const int n = pt.size();
  int is = 0, js = 0;
  for (int i = 1; i < n; ++i) {
    if (imag(pt[i]) > imag(pt[is])) is = i;
    if (imag(pt[i]) < imag(pt[js])) js = i;
  }
  double maxd = norm(pt[is]-pt[js]);

  int i, maxi, j, maxj;
  i = maxi = is;
  j = maxj = js;
  do {
    if (cross(diff(pt,i), diff(pt,j)) >= 0) j = (j+1) % n;
    else i = (i+1) % n;
    if (norm(pt[i]-pt[j]) > maxd) {
      maxd = norm(pt[i]-pt[j]);
      maxi = i; maxj = j;
    }
  } while (i != is || j != js);
  return maxd; /* farthest pair is (maxi, maxj). */
}

//最近点対 (求めるのは距離だけで、ペアの2点は求めない)
//多角形gで呼び出す時は、convex_radius(&g[0], N)みたいに配列になおしてください！
//また、gはあらかじめcompare_xでソートして下さい。
double compare_x(const P &a, const P &b){
  return a.X != b.X ? a.X < b.X : a.Y < b.Y;
}

double compare_y(const P &a, const P &b){
  return a.Y != b.Y ? a.Y < b.Y : a.X < b.X;
}

double convex_radius(P *a, int n){
  if(n <= 1) return INF;
  int m = n / 2;
  double x = a[m].X;
  double d = min(convex_radius(a, m), convex_radius(a + m, n - m));
  sort(a, a + n, compare_y);
  vector<P> b;
  for(int i = 0; i < n; ++i){
    if(fabs(a[i].X - x) >= d) continue;
    for(int j = 0; j < (int)b.size(); ++j){
      P p0 = a[i], p1 = b[(int)(b.size()) - j - 1];
      if(p0.Y - p1.Y >= d) break;
      d = min(d, abs(p0 - p1));
    }
    b.push_back(a[i]);
  }
  return d;
}

// 凸多角形の直線カット
// 凸多角形を直線で切断し、その左側(直線と交差する点も含む)だけを残す

G convex_cut(const G &g, const L &l){
  G Q;
  REP(i, g.size()){
    P A = curr(g, i), B = next(g, i);
    if(ccw(l[0], l[1], A) != -1) Q.push_back(A);
    if(ccw(l[0], l[1], A) * ccw(l[0], l[1], B) < 0) Q.push_back(crosspointLL(L(A, B), l));
  }
  return Q;
}

// ベクトルの回転
P rotation(P v, double r){
  return P(v.real() * cos(r) - v.imag() * sin(r), v.real() * sin(r) + v.imag() * cos(r));
}

//2ベクトル間の角度
atan2(x1 * y2 - x2 * y1, x1 * x2 + y1 * y2);
```
