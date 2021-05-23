#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Morris-Lecar model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import cv2

def kuramoto2d(N, T, t0, dt, s, K0, K1, sd_omega):
    s_sqrt_dt = s*np.sqrt(dt)
    omega = 1.0 + sd_omega * (np.random.rand(N,N) - 0.5)
    phi = 2*np.pi * np.random.rand(N,N) - np.pi
    zR = np.cos(phi)
    X = np.zeros((T,N,N))
    dphi = np.zeros((N,N))
    phi_up = np.zeros((N,N))
    phi_down = np.zeros((N,N))
    phi_left = np.zeros((N,N))
    phi_right = np.zeros((N,N))
    # coupling constant time course
    u = K0*np.ones(t0)
    v = np.linspace(K0,K1,num=T)
    K = np.hstack((u,v)) # 1-dim time course of K
    K4 = K/4
    # iterate
    for t in range(t0+T):
        if (t%100 == 0): print(f"    t = {t:d}/{t0+T:d}\r", end="")
        phi_up = np.roll(phi, -1, axis=0)
        phi_down = np.roll(phi, 1, axis=0)
        phi_left = np.roll(phi, -1, axis=1)
        phi_right = np.roll(phi, 1, axis=1)
        dphi = np.zeros((N,N))
        dphi += (np.sin(phi_up-phi) + np.sin(phi_down-phi))
        dphi += (np.sin(phi_left-phi) + np.sin(phi_right-phi))
        dphi = omega + K4[t]*dphi
        # stochastic integration
        phi += (dphi*dt + s_sqrt_dt*np.random.randn(N,N))
        if (t >= t0):
            #X[t-t0,:,:] = np.cos(phi)
            X[t-t0,:,:] = np.angle(np.exp(1j*phi)) # phase
    print("\n")
    return X


def animate_pyplot1(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.FuncAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/simple_anim.html
    (Faster than animate_pyplot2)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    vmin, vmax = data.min(), data.max()
    # setup animation image
    cm = plt.cm.gray # plt.cm.jet # plt.cm.gray
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    t = plt.imshow(data[0,:,:], origin="lower", cmap=cm,
                   interpolation="bilinear", vmin=vmin, vmax=vmax)
    plt.tight_layout()
    # frame generator
    print("[+] Animate")
    def animate(i):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        t.set_data(data[i,:,:])
    # create animation
    ani = animation.FuncAnimation(fig, animate, frames=n1, interval=10)
    #ani.save(fname)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_pyplot2(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.ArtistAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/dynamic_image.html
    (Slower than animate_pyplot1)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    print("[+] Animate")
    vmin, vmax = data.min(), data.max()
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    plt.tight_layout()
    ims = []
    for i in range(n1):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        im = ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                       vmin=vmin, vmax=vmax, animated=True)
        if i == 0:
            ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                           vmin=vmin, vmax=vmax)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    #ani.save(fname2)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_video(fname, x, downsample=10):
    nt, nx, ny = x.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, x, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        x = f_ip(t1)
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    #print(f"n1 = {n1:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 30
    print("[+] Animate")
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    for i in range(n1):
        print(f"    i = {i:d}/{n1:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def main():
    print("Kuramoto lattice model\n")
    N = 100 # 128
    T = 2500
    t0 = 500
    dt = 0.05
    s = 0.1
    # Kuramoto parameters
    K0 = 1.0
    K1 = 5.0
    sd_omega = 0.25
    print(f"[+] Lattice size N: {N:d}")
    print(f"[+] Time steps T: {T:d}")
    print(f"[+] Warm-up steps t0: {t0:d}")
    print(f"[+] Integration time step dt: {dt:.4f}")
    print(f"[+] Noise intensity: {s:.2f}")
    print(f"[+] Coupling constant: K0={K0:.2f} ... K1={K1:.2f}")
    print(f"[+] Frequency distribution std. dev.: {sd_omega:.3f}")

    # run simulation
    data = kuramoto2d(N, T, t0, dt, s, K0, K1, sd_omega)
    #print("[+] Data dimensions: ", data.shape)

    ''' plot mean signal
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.tight_layout()
    plt.show()
    '''

    # save data
    #fname1 = f"kuramoto2d_K_{K0:.2f}_{K1:.2f}_s_{s:.2f}.npy"
    #np.save(fname1, data)
    #print(f"[+] Data saved as: {fname1:s}")

    # video
    fname2 = f"kuramoto2d_K_{K0:.2f}_{K1:.2f}_s_{s:.2f}.mp4"
    #animate_pyplot1(fname2, data, downsample=5)
    #animate_pyplot2(fname2, data, downsample=5)
    animate_video(fname2, data, downsample=5) # fastest
    print(f"[+] Animation saved as: {fname2:s}")


if __name__ == "__main__":
    os.system("clear")
    main()
