#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Kuramoto model on a 2D lattice
# FvW 03/2018

using Interpolations
using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function kuramoto2d(N, T, t0, dt, s, K0, K1, sd_omega)
    s_sqrt_dt = s*sqrt(dt)
	omega = 1.0 .+ sd_omega * (rand(N,N) .- 0.5)
    phi = 2pi .* rand(N,N) .- pi
	X = zeros(Float64,T,N,N)
	dphi = zeros(Float64,N,N)
	phi_up = zeros(Float64,N,N)
	phi_down = zeros(Float64,N,N)
	phi_left = zeros(Float64,N,N)
    phi_right = zeros(Float64,N,N)
    # coupling constant time course
    u = K0*ones(t0)
    v = [_ for _ in range(K0,stop=K1,length=T)]
    K = vcat(u,v) # 1-dim time course of K
    K4 = K .* 0.25
    # iterate
    for t in range(1, stop=t0+T, step=1)
        (t%100 == 0) && print("    t = ", t, "/", t0+T, "\r")
		phi_up = circshift(phi, [-1 0])
        phi_down = circshift(phi, [1 0])
        phi_left = circshift(phi, [0 -1])
        phi_right = circshift(phi, [0 1])
        dphi = zeros(Float64,N,N)
	    dphi += (sin.(phi_up-phi) + sin.(phi_down-phi))
	    dphi += (sin.(phi_left-phi) + sin.(phi_right-phi))
        dphi = omega .+ K4[t] .* dphi
		# stochastic integration
		phi += (dphi*dt + s_sqrt_dt*randn(N,N))
		#(t > t0) && (X[t-t0,:,:] = cos.(phi))
        (t > t0) && (X[t-t0,:,:] = angle.(exp.(im.*phi))) # angle
    end
    println("\n")
    return X
end

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
			   vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    println("[+] animate")
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data, downsample=10)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    n1 = nt
    if (downsample > 0)
        println("[+] Downsampling ", downsample)
        n1 = Int(nt/downsample) # number of samples after downsampling
        data1 = zeros(Complex64,n1,nx,ny)
        t0 = range(1,stop=nt)
        t1 = range(1,stop=nt,length=n1)
        for i in range(1,stop=nx)
            for j in range(1,stop=ny)
                y = data[:,i,j]
                f_ip = LinearInterpolation(t0, y)
                data1[:,i,j] = f_ip(t1)
            end
        end
    else
        data1 = copy(data)
    end
    # BW
    y = UInt8.(round.(255*(data1 .- minimum(data1)) /
                           (maximum(data1)-minimum(data1))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data1 .- minimum(data1)) /
    #                              (maximum(data1)-minimum(data1))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=30
    T = size(data1,1)
    println("[+] Animate:")
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=n1,step=1)
            (i%100 == 0) && print("    i = ", i, "/", T, "\r")
            write(writer, y[i,end:-1:1,:])
        end
    end
    println("\n")
end

function main()
    println("Kuramoto lattice model\n")
    N = 128
    T = 3000
    t0 = 0
    dt = 0.05
    s = 0.2
    # Kuramoto parameters
    K0 = 0.5
    K1 = 4.0
    sd_omega = 0.25
    println("[+] Lattice size N: ", N)
    println("[+] Time steps T: ", T)
    println("[+] Warm-up steps t0: ", t0)
    println("[+] Integration time step dt: ", dt)
    println("[+] Noise intensity s: ", s)
	println("[+] Coupling constant K0...K1: ", K0, " ... ", K1)
    println("[+] Frequency distribution std. dev.: ", sd_omega)

    # run simulation
    data = kuramoto2d(N, T, t0, dt, s, K0, K1, sd_omega)
    println("[+] Data dimensions: ", size(data))

    # plot mean signal
    m = mean(reshape(data, (T,N*N)), dims=2)
    plot(m, "-k"); show()

    # save data
    K0_str = rpad(K0, 4, '0') # coupl. param. 4-char string
    K1_str = rpad(K1, 4, '0') # coupl. param. 4-char string
    s_str = rpad(s, 4, '0') # noise as 4-char string
    fname1 = string("kuramoto2d_K_", K0_str, "_", K1_str, "_s_", s_str, ".npy")
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", fname1)

	# video
    fname2 = string("kuramoto2d_K_", K0_str, "_", K1_str, "_s_", s_str, ".mp4")
    #animate_pyplot(fname2, data) # slow
    animate_video(fname2, data, 5) # fast
    println("[+] Data saved as: ", fname2)
end

main()
