Shader "Test/SurfaceOnly"
{
    Properties { _MainTex ("Albedo", 2D) = "white" {} }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma surface surf Standard fullforwardshadows
            #pragma target 3.0
            sampler2D _MainTex;
            struct Input { float2 uv_MainTex; };
            void surf(Input IN, inout SurfaceOutputStandard o) { o.Albedo = tex2D(_MainTex, IN.uv_MainTex).rgb; }
            ENDCG
        }
    }
}
