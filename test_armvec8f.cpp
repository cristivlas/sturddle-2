#include "common.h"
#include "armvector.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void testVec8fConstructorAndElementAccess()
{
    // Test value constructor
    Vec8f v1(1.0f);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(v1[i] == 1.0f);
    }

    // Test individual value constructor
    Vec8f v2(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    ASSERT_ALWAYS(v2[0] == 1.0f);
    ASSERT_ALWAYS(v2[1] == 2.0f);
    ASSERT_ALWAYS(v2[2] == 3.0f);
    ASSERT_ALWAYS(v2[3] == 4.0f);
    ASSERT_ALWAYS(v2[4] == 5.0f);
    ASSERT_ALWAYS(v2[5] == 6.0f);
    ASSERT_ALWAYS(v2[6] == 7.0f);
    ASSERT_ALWAYS(v2[7] == 8.0f);
}

void testVec8fOperations()
{
    // Test addition
    Vec8f a(1.0f);
    Vec8f b(2.0f);
    Vec8f c = a + b;
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(c[i] == 3.0f);
    }

    // Test subtraction
    c = a - b;
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(c[i] == -1.0f);
    }

    // Test multiplication
    c = a * b;
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(c[i] == 2.0f);
    }

    // Test max
    c = max(a, b);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(c[i] == 2.0f);
    }
}

void testHorizontalAdd()
{
    // Test horizontal add
    Vec8f v(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    float sum = horizontal_add(v);
    ASSERT_ALWAYS(sum == 36.0f); // 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
}

void testFusedMulAdd()
{
    // Test fused mul-add
    Vec8f a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    Vec8f b(2.0f);
    Vec8f c(3.0f);
    Vec8f d = mul_add(a, b, c);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(d[i] == a[i] * b[0] + c[0]); // 2*a[i] + 3
    }
}

void testLoadAndStore()
{
    // Test load
    float data1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Vec8f v;
    v.load_a(data1);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(v[i] == data1[i]);
    }

    // Test store
    Vec8f a(9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
    float data2[8];
    a.store_a(data2);
    for (int i = 0; i < 8; i++)
    {
        // std::clog << "i: " << data2[i] << " " << a[i] << std::endl;
        ASSERT_ALWAYS(data2[i] == a[i]);
    }
}

void testSingleValueConstructor()
{
    Vec8f v(5.0f);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(v[i] == 5.0f);
    }
}

void testVectorConstructor()
{
    float16x8_t f = vdupq_n_f16(3.0f);
    Vec8f v(f);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(v[i] == 3.0f);
    }
}

void testTwoVectorConstructor()
{
    float32x4_t f1 = vdupq_n_f32(2.0f);
    float32x4_t f2 = vdupq_n_f32(4.0f);
    Vec8f v(f1, f2);
    for (int i = 0; i < 4; i++)
    {
        ASSERT_ALWAYS(v[i] == 2.0f);
    }
    for (int i = 4; i < 8; i++)
    {
        ASSERT_ALWAYS(v[i] == 4.0f);
    }
}

void testEightFloatsConstructor()
{
    Vec8f v(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    for (int i = 0; i < 8; i++)
    {
        ASSERT_ALWAYS(v[i] == i + 1.0f);
    }
}

int main()
{
    testSingleValueConstructor();
    testVectorConstructor();
    testTwoVectorConstructor();
    testEightFloatsConstructor();
    testVec8fConstructorAndElementAccess();
    testVec8fOperations();
    testHorizontalAdd();
    testFusedMulAdd();
    testLoadAndStore();

    std::cout << "__ARM_FEATURE_FP16_VECTOR_ARITHMETIC ok\n";
}
#else
int main()
{
    std::cout << "__ARM_FEATURE_FP16_VECTOR_ARITHMETIC not supported\n";
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
