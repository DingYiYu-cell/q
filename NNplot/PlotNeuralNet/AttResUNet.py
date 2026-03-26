arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    # 输入层
    to_input('../examples/fcn8s/cats.jpg'),
    
    # 编码器部分
    to_ConvConvRelu(name='ccr_b1', s_filer=500, n_filer=(64,64), 
                   offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", 
           width=1, height=32, depth=32, opacity=0.5),
    
    # 下采样模块（重复3次）
    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', 
                    s_filer=256, n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5),
    
    # 瓶颈层
    to_ConvConvRelu(name='ccr_b5', s_filer=32, n_filer=(1024,1024), 
                   offset="(2,0,0)", to="(pool_b4-east)", width=(8,8), height=8, depth=8, caption="Bottleneck"),
    
    # 解码器部分（上采样+跳跃连接）
    *block_Unconv(name="b6", botton="ccr_b5", top='end_b6', 
                 s_filer=64, n_filer=512, offset="(2.1,0,0)", size=(16,16,5.0), opacity=0.5),
    to_skip(of='ccr_b4', to='ccr_res_b6', pos=1.25),
    
    # 输出层
    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.75,0,0)", 
                  to="(end_b9-east)", width=1, height=40, depth=40, caption="SOFT"),
    to_end() 
]