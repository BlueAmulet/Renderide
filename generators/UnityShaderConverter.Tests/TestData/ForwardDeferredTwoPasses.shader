// Test: one forward and one deferred pass; filtering should drop deferred only.
Shader "Converter/ForwardDeferredTwoPasses"
{
    Properties { _Color ("Color", Color) = (1,1,1,1) }
    SubShader
    {
        Pass
        {
            Tags { "LightMode" = "ForwardBase" }
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            struct appdata { float4 vertex : POSITION; };
            struct v2f { float4 vertex : SV_POSITION; };
            v2f vert (appdata v) { v2f o; o.vertex = v.vertex; return o; }
            float4 frag (v2f i) : SV_Target { return float4(1,0,0,1); }
            ENDCG
        }
        Pass
        {
            Tags { "LightMode" = "Deferred" }
            CGPROGRAM
            #pragma vertex vert2
            #pragma fragment frag2
            struct appdata2 { float4 vertex : POSITION; };
            struct v2f2 { float4 vertex : SV_POSITION; };
            v2f2 vert2 (appdata2 v) { v2f2 o; o.vertex = v.vertex; return o; }
            float4 frag2 (v2f2 i) : SV_Target { return float4(0,1,0,1); }
            ENDCG
        }
    }
}
