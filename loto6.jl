using DataFrames,CSV,Dates
using Flux,Statistics

using ProgressMeter
using StatsBase

work_path= @__DIR__
dl_url = "https://loto6.thekyo.jp/data/loto6.csv"
data_csv = joinpath(work_path , "loto6dl.csv")
wide = 6

println("***CHECK CSV DATA")

function dataupdate(dl_url, data_csv)
    if isfile(data_csv)
        Base.download(dl_url,data_csv)
    else
        fs=stat(data_csv)
        dt = unix2datetime(fs.mtime)
        #ddt = Date(dt)
        td = now()
        #dtd = Date(td)
        if (Date(td) - Date(dt)) > Day(1)
            Base.download(dl_url,data_csv)
            println("***UPDATE DF DATA")
        end
    end
end

dataupdate(dl_url,data_csv)

#df = DataFrame!(CSV.File(data_csv,threaded=false))

function getdatamatrix(data_csv)
    df=CSV.read(data_csv,DataFrame)
    m,n = size(df)

    datamx = df[:,3:9]|> Matrix #１桁～Bonous数字まで
    return datamx
end

function transformtozeroone(da ; E=43)
    m,n=1,0
    if typeof(da) == Matrix{Int64}
        m, n = size(da)
    else
        m, n = 1, length(da)
    end
    rs = zeros(Float64,E)
    for d in da
        _d = Flux.onehot(d,1:E)
        rs .= rs .+ _d
    end
    return rs = rs ./ m
end 

function transformys(da ; E=43)
    rs = zeros(Int64,E)
    n = length(da)
    rs = [Flux.onehot(e,1:E) for e in da] |> t->hcat(t...)
    return rs
end 

function getxydata(datamx ; wide=wide, m=size(datamx,1))
    xs = [ datamx[i:(i+wide-1),:] |> transformtozeroone for i = 1:(m-wide)] |> t -> hcat(t...)
    ys = [ datamx[i+1,:]  |> transformtozeroone for i = 1:(m-wide) ] |> t -> hcat(t...)
    return (xs,ys)
end


function getmodel()
    model = Chain(
        Dense(43 => 43, tanh), Dense(43 => 43, tanh),Dense(43 => 43, tanh),
        Dense(43 => 43, tanh),Dense(43 => 43, tanh), Dense(43 => 43, tanh), 
        Dense(43 => 43, tanh),Dense(43 => 43, tanh), Dense(43 => 43, tanh), 
        Dense(43 => 43, tanh),Dense(43 => 43, tanh), Dense(43 => 43, tanh), 
        Dense(43 => 43), softmax)
    end
    
    
    
function inference(loss_f::Function, model,loader,(xs,ys))
    optim = Flux.setup(Flux.Adam(0.01), model)
    losses=[]
    @showprogress for epoch in 1:50
        for (x,y) in loader
            #        @show size(y) size(x)
            loss, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                loss_f(x,y_hat)
            end
            Flux.update!(optim,model, grads[1])
            push!(losses, loss)
        end
    end
    return losses,model
end
    #plot(losses)
    
function lotopredict(mmodel, datamx, ; w=wide)
    m,n=size(datamx)
    lastxs = datamx[(m-w+1):m ,:] |> transformtozeroone
    mmodel(lastxs)
end

function seeloto(wgt,n)
    rs = []
    for i = 1:n
        _r = sample(1:43, wgt, 6, ; replace=false) |> sort!
        push!(rs,_r)
    end
    return rs
end
    
function printout(wgt, ; n=10)
    rs = seeloto(wgt, n)
    for e in rs
        println(e)
    end
end

datamx = getdatamatrix(data_csv)
xs,ys = getxydata(datamx)
loader = Flux.DataLoader((xs,ys),batchsize=64, shuffle=true)
model=getmodel()
loss_func(x,y) = ( x .- y ) .^2 |> mean 
losses, model_update = inference(loss_func, model, loader, (xs,ys))
wgt = lotopredict(model_update,datamx) |> ProbabilityWeights
printout(wgt)