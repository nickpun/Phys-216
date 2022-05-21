##########################
# ballistics
# Aaron Boley
# Revision 2019-03-05
##########################
import numpy as np
import matplotlib.pylab as plt
import sys


def main(alt=80.,az=180.,dt=0.1,H0=4.,V0=10.0,mrkt=5.,mfuel0=5.,CD=0.0025,
         T0=0.0,g=9.8,HSCALE=8e3,REARTH=6370e3,plot='T',integrator='pc',
         lat=49.,om=7.29e-5,ve=2000,mrate=0.25): #!!!
    '''
    Calculate solutions for HW4 problem 3.  Options are
    alt [45.]           altitude of initial release at V0. degrees.
    az  [90.]           azimuth of initial release at V0. degrees.
    dt [1.0]            time step in s  
    H0 [4]              starting height in m
    V0 [10]             starting speed in m/s
    mrkt [5]            mass of rocket in kg
    mfuel0 [5]          initial mass of fuel in kg
    T0 [0]              starting time in s
    g [9.8]             gravity at z=0 in m/s/s
    HSCALE [8e3]        scaleheight of atmosphere in m
    REARTH [6370e3]     radius of Earth
    plot [T]            Plot the arrays ('T' or 'F').  Must be passed as a character!!!
    integrator [euler]  This is the integrator used for the problem. Choices are 'euler' and 'pc' (predictor-corrector)
    ve [2000]           exhaust velocity in m/s
    mrate               burn rate of the fuel in kg/s

    import into python as projectile.  For example "import projectile as pj"
    
    Run using something like "pj.main(dt=0.1)"

    For help after import, type help(pj)
    '''

    alt*=np.pi/180.
    az*=np.pi/180.
    lat*=np.pi/180.

    def mag(v): 
        '''v is a general vector in this case'''
        return    np.sqrt(  v[0]*v[0]+ v[1]*v[1]+  v[2]*v[2] )

    def grav(z): return -g / (1 + (z/REARTH))**2

    def acceleration(r,v,m):
        a=np.zeros(3)
        vmag=mag(v)
        expon=-CD*np.exp(-r[2]/HSCALE)*vmag/m
        a[0]=expon*v[0]
        a[1]=expon*v[1]
        a[2]=expon*v[2] + grav(r[2])
        return a


    def Rx(v,a):
        '''v is a general vector in this case'''
        s=np.zeros(3)
        s[0] = v[0]
        s[1] = np.cos(a)*v[1] -np.sin(a)*v[2]
        s[2] = np.sin(a)*v[1] +np.cos(a)*v[2]
        return s

    def Rz(v,a):
        '''v is a general vector in this case'''
        s=np.zeros(3)
        s[0] = np.cos(a)*v[0] - np.sin(a)*v[1]
        s[1] = np.sin(a)*v[0] + np.cos(a)*v[1]
        s[2] = v[2]
        return s

    def centrifugal(om,r,l):
        cosl=np.cos(l)
        sinl=np.sin(l)
        a=np.zeros(3)
        om2=om*om
 
        a[0] = om2*r[0]
        a[1] = -om2*(r[2]*sinl*cosl-r[1]*sinl**2)
        a[2] = -om2*(r[1]*cosl*sinl-r[2]*cosl**2)
        return a

    def coriolis(om,v,l):
        cosl=np.cos(l)
        sinl=np.sin(l)
        a=np.zeros(3)

        a[0]=-2*om*(v[2]*cosl-v[1]*sinl)
        a[1]=-2*om*v[0]*sinl
        a[2]=2*om*v[0]*cosl
        return a
    
    def inertial(om,l): #!!!
        cosl=np.cos(l)
        sinl=np.sin(l)
        a=np.zeros(3)
        om2=om*om
        
        a[0] = 0
        a[1] = -om2*REARTH*cosl*sinl
        a[2] = om2*REARTH*cosl*cosl
        return a
    
    def thrust(t,v): #!!!
        '''v is a general vector in this case'''
        a = np.zeros(3)
        if t < mfuel0/mrate:
            acc = mrate*ve/(mass(t))
        
            a[0] = acc*v[0]/np.sum(v)
            a[1] = acc*v[1]/np.sum(v)
            a[2] = acc*v[2]/np.sum(v)
        return a
    
    def mass(t): #!!!
        m = mrkt
        if t < mfuel0/mrate:
            m += mfuel0 - mrate*t
        return m


    v=np.array([0.,V0,0.])
    v=Rx(v,alt)
    v=Rz(v,-az) # negative because azimuth is measured eastward from north (clockwise)

    it=0
    rarr=np.array([[0.,0.,H0]])
    varr=np.array([v])
    tarr=np.array(T0)

    r=np.array([0.,0.,H0])
    t=T0

    aarr=np.array([acceleration(r,v,mass(t))]) #!!!
    aarr+=coriolis(om,v,lat)+centrifugal(om,r,lat)
    aarr+=inertial(om,lat) #!!!
    aarr+=thrust(t,v) #!!!

    if integrator=='euler':

      while r[2]>0 or it==0:
        a = acceleration(r,v,mass(t)) #!!!
        a += coriolis(om,v,lat)+centrifugal(om,r,lat)
        a += inertial(om,lat) #!!!
        a += thrust(t,v) #!!!
        for i in range(3): 
            r[i]+=v[i]*dt
            v[i]+=a[i]*dt
        t+=dt
        rarr=np.append(rarr,[r],axis=0) 
        varr=np.append(varr,[v],axis=0)
        aarr=np.append(aarr,[a],axis=0)
        tarr=np.append(tarr,t)
        it+=1

    elif integrator=='pc':

      rp=np.zeros(3)
      vp=np.zeros(3)
      rc=np.zeros(3)
      vc=np.zeros(3)
      rm=np.zeros(3)
      vm=np.zeros(3)
      while r[2]>0 or it==0:
        # first take trial step
        a = acceleration(r,v,mass(t)) #!!!
        a += coriolis(om,v,lat)+centrifugal(om,r,lat)
        a += inertial(om,lat) #!!!
        a += thrust(t,v) #!!!
        for i in range(3):
            rp[i]=r[i]+v[i]*dt
            vp[i]=v[i]+a[i]*dt
    
        # now add a corrector, which is just an estimate using the acceleration at the end of the trial step
        ap = acceleration(rp,vp,mass(t)) #!!!
        ap += coriolis(om,vp,lat)+centrifugal(om,rp,lat)
        ap += inertial(om,lat) #!!!
        ap += thrust(t,vp) #!!!
        for i in range(3):
            rc[i] = r[i]+vp[i]*dt
            vc[i] = v[i]+ap[i]*dt

        ac = acceleration(rc,vc,mass(t)) # this step is mainly for record keeping #!!!
        ac+= coriolis(om,vc,lat)+centrifugal(om,rc,lat)
        ac+= inertial(om,lat) #!!!
        ac+= thrust(t,vc) #!!!
        # now average the solutions for final z estimate

        # this is just a shortcut to ensure the new array is not just a pointer to the old array
        r0=r*1
        v0=v*1

        r = 0.5*(rp+rc)
        v = 0.5*(vp+vc)
        a = 0.5*(ap+ac)

        t+=dt

        rarr=np.append(rarr,[r],axis=0) 
        varr=np.append(varr,[v],axis=0)
        aarr=np.append(aarr,[a],axis=0)
        tarr=np.append(tarr,t)
        it+=1

    else:
        return "Invalid integrator selection"

    #interpolate
    # z = zm + (zp - zm)/(tp-tm) dt

    deltat = -rarr[it-1][2] / varr[it-1][2]  # delta t for hitting the ground

    for i in range(3):
        rarr[it][i]=rarr[it-1][i] + varr[it-1][i]*deltat  # recalculate to check
        varr[it][i]=varr[it-1][i]+aarr[it-1][i]*deltat
    aarr[it]=acceleration(rarr[it],varr[it],mass(t)) #!!!
    aarr[it]+=coriolis(om,varr[it],lat)+centrifugal(om,rarr[it],lat)
    aarr[it]+=inertial(om,lat) #!!!
    aarr[it]+=thrust(t,varr[it]) #!!!
    tarr[it]=tarr[it-1]+deltat

    print("Hit the ground at T = {} s with vx,vy,vz = {} m/s and x,y,z = {} m".format(tarr[it],varr[it],rarr[it]))

    if plot=="T":

      R=np.zeros(len(rarr[:,0]))

      phi=np.arctan2(rarr[:,1],rarr[:,0])
      R=np.sqrt(rarr[:,0]**2+rarr[:,1]**2)
      phi[0]=phi[1]
      drange = R*(phi-phi[0]) 

      plt.figure()
      plt.xlabel('Time [s]')
      plt.ylabel('V [m/s]')
      plt.plot(tarr,varr[:,2])

      plt.figure()
      plt.xlabel('Time [s]')
      plt.ylabel('Z [m]')
      plt.plot(tarr,rarr[:,2])

      plt.figure()
      plt.xlabel('R [m]')
      plt.ylabel('Z [m]')
      plt.plot(np.sqrt(rarr[:,0]**2+rarr[:,1]**2),rarr[:,2])


      plt.figure()
      plt.xlabel('t [s]')
      plt.ylabel('a [m/s/s]')
      plt.plot(tarr,aarr[:,2])

      plt.figure()
      plt.xlabel('VR [m/s]')
      plt.ylabel('VZ [m/s]')
      plt.plot(np.sqrt(varr[:,0]**2+varr[:,1]**2),varr[:,2])

      plt.figure()
      plt.xlabel('X [m]')
      plt.ylabel('Y [m]')
      plt.plot(rarr[:,0],rarr[:,1])

      plt.figure()
      plt.xlabel('R [m]')
      plt.ylabel('dR [m]')
      plt.plot(R,drange)


      plt.show()


if __name__=='__main__':main()
  


