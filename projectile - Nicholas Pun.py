import numpy as np
import matplotlib.pylab as plt
import sys


def main(omega=7.29e-5,lam=49,alt=19.46,az=100.08,dt=0.0001,H0=4.,V0=200.0,mass=10.,bdrag=0.043,T0=0.0,g=9.8,HSCALE=8e3,REARTH=6370e3,plot='T',integrator='pc'):
    '''
    Calculate solutions for HW1 problem 3.  Options are
    omega
    alt [45.]     altitude of initial release at V0. degrees.
    az  [90.]     azimuth of initial release at V0. degrees.
    dt [0.1]      time step in s  
    H0 [30e3]     starting height in m
    V0 [0]        starting speed in m/s
    bdrag [0.5]   the drag term in kg/m
    mass [100]    mass in kg
    T0 [0]        starting time in s
    g [9.8]       gravity at z=0 in m/s/s
    HSCALE [8e3]  scaleheight of atmosphere in m
    REARTH [6370e3] radius of Earth
    plot [F]      Plot the arrays ('T' or 'F').  Must be passed as a character!!!
    integrator [pc]  This is the integrator used for the problem. Choices are 'euler' and 'pc' (predictor-corrector)

    import into python as projectile.  For example "import projectile as pj"
    
    Run using something like "pj.main(dt=0.1)"

    For help after import, type help(pj)
    '''

    alt*=np.pi/180.
    az*=np.pi/180.

    def mag(v): 
        '''v is a general vector in this case'''
        return    np.sqrt(  v[0]*v[0]+ v[1]*v[1]+  v[2]*v[2] )

    def grav(z): return -g / (1 + (z/REARTH))**2

    def acceleration(r,v,mass):
        a=np.zeros(3)
        vmag=mag(v)
        expon=-bdrag*np.exp(-r[2]/HSCALE)*vmag/mass
        a[0]=expon*v[0] +2*omega*(v[1]*np.sin(lam)-v[2]*np.cos(lam)+g*np.cos(lam))
        a[1]=expon*v[1] -2*omega*v[0]*np.sin(lam)
        a[2]=expon*v[2] + grav(r[2]) -g+2*omega*v[0]*np.cos(lam)
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


    v=np.array([0.,V0,0.]) # set initial velocity perfectly in y direction and then rotate
    v=Rx(v,alt)
    v=Rz(v,-az) # negative because azimuth is measured eastward from north (clockwise)

    it=0
    rarr=np.array([[0.,0.,H0]])
    varr=np.array([v])
    tarr=np.array(T0)

    r=np.array([0.,0.,H0])
    t=T0

    aarr=np.array([acceleration(r,v,mass)])

    if integrator=='euler':

      while r[2]>0 or it==0:
        a = acceleration(r,v,mass) 
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
        a = acceleration(r,v,mass)
        for i in range(3):
            rp[i]=r[i]+v[i]*dt
            vp[i]=v[i]+a[i]*dt
    
        # now add a corrector, which is just an estimate using the acceleration at the end of the trial step
        ap = acceleration(rp,vp,mass)
        for i in range(3):
            rc[i] = r[i]+vp[i]*dt
            vc[i] = v[i]+ap[i]*dt

        ac = acceleration(rc,vc,mass) # this step is mainly for record keeping
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
    aarr[it]=acceleration(rarr[it],varr[it],mass)
    tarr[it]=tarr[it-1]+deltat

    print("Hit the ground at T = {} s with vx,vy,vz = {} m/s and x,y,z = {} m".format(tarr[it],varr[it],rarr[it]))

    if plot=="T":

#      plt.figure()
#      plt.xlabel('Time [s]')
#      plt.ylabel('V [m/s]')
#      plt.plot(tarr,varr[:,2])

#      plt.figure()
#      plt.xlabel('Time [s]')
#      plt.ylabel('Z [m]')
#      plt.plot(tarr,rarr[:,2])

      plt.figure()
      plt.xlabel('R [m]')
      plt.ylabel('Z [m]')
      plt.title('Range and Height')
      plt.plot(np.sqrt(rarr[:,0]**2+rarr[:,1]**2),rarr[:,2])
      
      plt.figure()
      plt.xlabel('x prime [m]')
      plt.ylabel('y prime [m]')
      plt.title('x prime - y prime path')
      plt.plot(rarr[:,0],rarr[:,1])


      plt.figure()
      plt.xlabel('t [s]')
      plt.ylabel('a [m/s/s]')
      plt.title('Acceleration as a function of time')
      plt.plot(tarr,aarr[:,2])

#      plt.figure()
#      plt.xlabel('VR [m/s]')
#      plt.ylabel('VZ [m/s]')
#      plt.plot(np.sqrt(varr[:,0]**2+varr[:,1]**2),varr[:,2])


      plt.show()


if __name__=='__main__':main()
  


