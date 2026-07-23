import numpy as np
SX=np.array([[0,1],[1,0]],dtype=complex)
SY=np.array([[0,-1j],[1j,0]],dtype=complex)
SZ=np.array([[1,0],[0,-1]],dtype=complex)
I2=np.eye(2,dtype=complex)
def bloch_to_rho(r):
 r=np.asarray(r,dtype=float)
 return 0.5*(I2+r[0]*SX+r[1]*SY+r[2]*SZ)
def rho_to_bloch(rho):
 rho=np.asarray(rho,dtype=complex)
 return np.array([np.real(np.trace(rho@SX)),np.real(np.trace(rho@SY)),np.real(np.trace(rho@SZ))],dtype=float)
def fidelity(a,b):
 w,v=np.linalg.eigh(a)
 w=np.maximum(w,0)
 s=v@np.diag(np.sqrt(w))@v.conj().T
 M=s@b@s
 w,_=np.linalg.eigh(M)
 return float(np.sum(np.sqrt(np.maximum(w,0)))**2)
def bures_distance(a,b):
 F=fidelity(a,b)
 F=np.clip(F,0,1)
 return np.sqrt(2-2*np.sqrt(F))
class MeasurementSimulator:
 def __init__(self,shots=1024,seed=None):
  self.shots=shots
  self.rng=np.random.default_rng(seed)
 def _r(self):
  a=self.rng.standard_normal()+1j*self.rng.standard_normal()
  b=self.rng.standard_normal()+1j*self.rng.standard_normal()
  n=np.sqrt(abs(a)**2+abs(b)**2)
  a,b=a/n,b/n
  U=np.array([[a,-b.conjugate()],[b,a.conjugate()]],dtype=complex)
  assert np.allclose(U@U.conj().T,I2)
  return U
 def random_basis(self):return self._r()
 def measure(self,rho,basis):
  rho=np.asarray(rho,dtype=complex)
  basis=np.asarray(basis,dtype=complex)
  p=np.zeros(2,dtype=float)
  for i in range(2):
   b=basis[:,i]
   p[i]=np.real(b.conj()@rho@b)
  p=np.maximum(p,0)
  p=p/np.sum(p)
  return self.rng.multinomial(self.shots,p),p
 def measure_random_bases(self,rho,n):
  B=[self.random_basis()for _ in range(n)]
  C,P=[],[]
  for b in B:
   c,p=self.measure(rho,b)
   C.append(c);P.append(p)
  return{'bases':B,'counts':C,'probs':P,'shots':self.shots}
class StateReconstructor:
 def __init__(self,method='mle',max_iter=500,tol=1e-10):
  assert method in('linear','mle')
  self.method=method;self.max_iter=max_iter;self.tol=tol
 def _p(self,r):
  r=np.asarray(r,dtype=float)
  n=np.linalg.norm(r)
  return r/n if n>1.0 else r
 def linear_inversion(self,d):
  A,y=[],[]
  for basis,count in zip(d['bases'],d['counts']):
   for i in range(2):
    b=basis[:,i]
    n=np.array([np.real(b.conj()@SX@b),np.real(b.conj()@SY@b),np.real(b.conj()@SZ@b)],dtype=float)
    A.append(n)
    y.append(2.0*count[i]/d['shots']-1.0)
  r,_,_,_=np.linalg.lstsq(np.array(A),np.array(y),rcond=None)
  return self._p(r)
 def maximum_likelihood(self,d):
  r=self.linear_inversion(d)
  bb=[]
  for basis in d['bases']:
   bbl=[]
   for i in range(2):
    b=basis[:,i]
    n=np.array([np.real(b.conj()@SX@b),np.real(b.conj()@SY@b),np.real(b.conj()@SZ@b)],dtype=float)
    bbl.append(n)
   bb.append(bbl)
  def ll(rv):
   s=0.0
   for bbl,count in zip(bb,d['counts']):
    for i in range(2):
     P=np.clip(0.5*(1.0+np.dot(rv,bbl[i])),1e-12,1.0)
     s+=count[i]*np.log(P)
   return s
  def grad(rv):
   g=np.zeros(3)
   for bbl,count in zip(bb,d['counts']):
    for i in range(2):
     P=np.clip(0.5*(1.0+np.dot(rv,bbl[i])),1e-12,1.0)
     g+=count[i]/P*0.5*bbl[i]
   return g
  best_r,best_ll,step=r.copy(),ll(r),0.1
  for _ in range(self.max_iter):
   g=grad(r)
   gn=np.linalg.norm(g)
   if gn<1e-15:break
   g=g/gn
   improved=False
   for _ in range(20):
    rn=self._p(r+step*g)
    ln=ll(rn)
    if ln>best_ll+self.tol:
     best_ll,best_r,improved=ln,rn.copy(),True
     step=min(step*1.5,1.0)
     break
    step*=0.5
   if not improved:break
   r=best_r.copy()
  return best_r
 def reconstruct(self,d):
  return self.linear_inversion(d)if self.method=='linear'else self.maximum_likelihood(d)
 def reconstruction_fidelity(self,d,true_rho):
  return fidelity(bloch_to_rho(self.reconstruct(d)),true_rho)
 def reconstruction_error(self,d,true_rho):
  return bures_distance(bloch_to_rho(self.reconstruct(d)),true_rho)
class TomographicTrajectoryGenerator:
 def __init__(self,base_generator,shots=1024,n_bases=6,reconstructor_method='mle',seed=None):
  self.base_generator=base_generator
  self.sim=MeasurementSimulator(shots=shots,seed=seed)
  self.rec=StateReconstructor(method=reconstructor_method)
  self.n_bases=n_bases
  self.shots=shots
 def generate_trajectory(self,n_steps=10):
  axis,w,g=self.base_generator.sample_physics()
  traj=self.base_generator.trajectory(n_steps,axis,w,g)
  exact_states=[bloch_to_rho(r) for r in traj]
  reconstructed,errors,fids,raw_data=[],[],[],[]
  for rho in exact_states:
   data=self.sim.measure_random_bases(rho,self.n_bases)
   r_rec=self.rec.reconstruct(data)
   reconstructed.append(r_rec)
   errors.append(bures_distance(bloch_to_rho(r_rec),rho))
   fids.append(fidelity(bloch_to_rho(r_rec),rho))
   raw_data.append(data)
  return{'exact_states':exact_states,'reconstructed_states':reconstructed,'reconstruction_errors':errors,'fidelities':fids,'measurement_data':raw_data,'shots':self.shots,'n_bases':self.n_bases}
 def generate_batch(self,n_trajectories,n_steps=10):
  return[self.generate_trajectory(n_steps)for _ in range(n_trajectories)]
def _t1():
 for _ in range(100):
  r=np.random.randn(3);r=r/np.linalg.norm(r)*np.random.random()
  rho=bloch_to_rho(r)
  assert np.allclose(r,rho_to_bloch(rho),atol=1e-12)
  assert np.allclose(rho,rho.conj().T,atol=1e-12)
  assert np.isclose(np.trace(rho),1.0,atol=1e-12)
  assert np.all(np.linalg.eigvalsh(rho)>=-1e-12)
 print("  [PASS] Bloch roundtrip")
def _t2():
 for _ in range(20):
  r=np.random.randn(3);r=r/np.linalg.norm(r)*np.random.random()
  assert np.isclose(fidelity(bloch_to_rho(r),bloch_to_rho(r)),1.0,atol=1e-10)
 p0,p1=np.array([1,0],dtype=complex),np.array([0,1],dtype=complex)
 assert np.isclose(fidelity(np.outer(p0,p0.conj()),np.outer(p1,p1.conj())),0.0,atol=1e-10)
 for _ in range(20):
  r1=np.random.randn(3);r1=r1/np.linalg.norm(r1)*np.random.random()
  r2=np.random.randn(3);r2=r2/np.linalg.norm(r2)*np.random.random()
  a,b=bloch_to_rho(r1),bloch_to_rho(r2)
  assert np.isclose(fidelity(a,b),fidelity(b,a),atol=1e-10)
 print("  [PASS] Fidelity properties")
def _t3():
 sim=MeasurementSimulator(shots=100000,seed=42)
 c,p=sim.measure(bloch_to_rho(np.array([0,0,1])),np.eye(2,dtype=complex))
 assert np.isclose(p[0],1.0,atol=1e-12)and np.isclose(p[1],0.0,atol=1e-12)
 c,p=sim.measure(bloch_to_rho(np.array([1,0,0])),np.eye(2,dtype=complex))
 assert np.isclose(p[0],0.5,atol=1e-12)and np.isclose(p[1],0.5,atol=1e-12)
 sim2=MeasurementSimulator(shots=500000,seed=123)
 c,p=sim2.measure(bloch_to_rho(np.array([0.3,-0.4,0.5])),sim2.random_basis())
 assert np.allclose(c/sim2.shots,p,atol=0.01)
 print("  [PASS] Measurement probabilities")
def _t4():
 bx=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
 by=np.array([[1,1],[1j,-1j]],dtype=complex)/np.sqrt(2)
 bz=np.eye(2,dtype=complex)
 rec=StateReconstructor(method='linear')
 for _ in range(50):
  r=np.random.randn(3);r=r/np.linalg.norm(r)*np.random.random()
  rho=bloch_to_rho(r)
  sim=MeasurementSimulator(shots=100000,seed=None)
  d={'bases':[bx,by,bz],'counts':[],'probs':[],'shots':sim.shots}
  for b in[bx,by,bz]:
   c,p=sim.measure(rho,b);d['counts'].append(c);d['probs'].append(p)
  assert np.allclose(rec.linear_inversion(d),r,atol=0.02)
 print("  [PASS] Linear inversion")
def _t5():
 rl=StateReconstructor(method='linear')
 rm=StateReconstructor(method='mle')
 sim=MeasurementSimulator(shots=512,seed=42)
 n=0
 for _ in range(100):
  r=np.random.randn(3);r=r/np.linalg.norm(r)*np.random.random()
  rho=bloch_to_rho(r)
  d=sim.measure_random_bases(rho,6)
  r_mle=rm.reconstruct(d)
  assert np.linalg.norm(r_mle)<=1.0+1e-10
  if bures_distance(bloch_to_rho(r_mle),rho)<=bures_distance(bloch_to_rho(rl.reconstruct(d)),rho)+1e-10:n+=1
 assert n>=40
 print(f"  [PASS] MLE vs linear: {n}/100")
def _t6():
 r=np.array([0.3,-0.5,0.7]);rho=bloch_to_rho(r)
 e=[]
 for shots in[64,128,256,512,1024,2048]:
  sim=MeasurementSimulator(shots=shots,seed=42)
  rec=StateReconstructor(method='mle')
  e.append(rec.reconstruction_error(sim.measure_random_bases(rho,6),rho))
 s=(np.log(e[-1])-np.log(e[0]))/(np.log(2048)-np.log(64))
 assert -0.7<=s<=-0.3
 print(f"  [PASS] Scaling: slope={s:.3f}")
def _t7():
 rec=StateReconstructor(method='linear')
 assert np.allclose(rec._p(np.array([2.0,0,0])),[1.0,0,0],atol=1e-12)
 assert np.linalg.norm(rec._p(np.array([2.0,0,0])))<=1.0+1e-12
 assert np.allclose(rec._p(np.array([0.3,0.4,0.2])),[0.3,0.4,0.2],atol=1e-12)
 print("  [PASS] Physicality projection")
def run_all_tests():
 print("="*60)
 print("FINITE-SHOT TOMOGRAPHY — SELF-TEST SUITE")
 print("="*60)
 tests=[_t1,_t2,_t3,_t4,_t5,_t6,_t7]
 p=0
 for t in tests:
  try:t();p+=1
  except AssertionError as e:print(f"  [FAIL] {t.__name__}:{e}")
  except Exception as e:print(f"  [ERR ] {t.__name__}:{e}")
 print(f"RESULT: {p}/{len(tests)} suites passed")
 print("="*60)
 return p==len(tests)
if __name__=="__main__":
 import sys
 sys.exit(0 if run_all_tests() else 1)
