use crate::core::{ColourModel, ImageBase};
use ndarray::{Data, DataOwned};
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::marker::PhantomData;

impl<A, T, C> Serialize for ImageBase<T, C>
where
    A: Serialize,
    T: Data<Elem = A>,
    C: ColourModel,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ImageBase", 2)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("model", C::NAME)?;
        state.end()
    }
}

struct ImageBaseVisitor<T, C> {
    _marker_a: PhantomData<T>,
    _marker_b: PhantomData<C>,
}

enum ImageBaseField {
    Data,
    Model,
}

impl<T, C> ImageBaseVisitor<T, C> {
    pub fn new() -> Self {
        ImageBaseVisitor {
            _marker_a: PhantomData,
            _marker_b: PhantomData,
        }
    }
}

static IMAGE_BASE_FIELDS: &[&str] = &["data", "model"];

impl<'de, A, T, C> Deserialize<'de> for ImageBase<T, C>
where
    A: Deserialize<'de>,
    T: DataOwned<Elem = A>,
    C: ColourModel,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("ImageBase", IMAGE_BASE_FIELDS, ImageBaseVisitor::new())
    }
}

impl<'de> Deserialize<'de> for ImageBaseField {
    fn deserialize<D>(deserializer: D) -> Result<ImageBaseField, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ImageBaseFieldVisitor;

        impl<'de> de::Visitor<'de> for ImageBaseFieldVisitor {
            type Value = ImageBaseField;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(r#""data" or "model""#)
            }

            fn visit_str<E>(self, value: &str) -> Result<ImageBaseField, E>
            where
                E: de::Error,
            {
                match value {
                    "data" => Ok(ImageBaseField::Data),
                    "model" => Ok(ImageBaseField::Model),
                    other => Err(de::Error::unknown_field(other, IMAGE_BASE_FIELDS)),
                }
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<ImageBaseField, E>
            where
                E: de::Error,
            {
                match value {
                    b"data" => Ok(ImageBaseField::Data),
                    b"model" => Ok(ImageBaseField::Model),
                    other => Err(de::Error::unknown_field(
                        &format!("{:?}", other),
                        IMAGE_BASE_FIELDS,
                    )),
                }
            }
        }

        deserializer.deserialize_identifier(ImageBaseFieldVisitor)
    }
}

impl<'de, A, T, C> de::Visitor<'de> for ImageBaseVisitor<T, C>
where
    A: Deserialize<'de>,
    T: DataOwned<Elem = A>,
    C: ColourModel,
{
    type Value = ImageBase<T, C>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ndarray-vision Image representation")
    }

    fn visit_seq<V>(self, mut visitor: V) -> Result<Self::Value, V::Error>
    where
        V: de::SeqAccess<'de>,
    {
        let data = visitor
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let model = visitor
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        Ok(ImageBase { data, model })
    }

    fn visit_map<V>(self, mut visitor: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut data = None;
        let mut model: Option<&str> = None;
        while let Some(key) = visitor.next_key()? {
            match key {
                ImageBaseField::Data => {
                    if data.is_some() {
                        return Err(de::Error::duplicate_field("data"));
                    }
                    data = Some(visitor.next_value()?);
                }
                ImageBaseField::Model => {
                    if model.is_some() {
                        return Err(de::Error::duplicate_field("model"));
                    }
                    model = Some(visitor.next_value()?);
                }
            }
        }
        let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
        let model = model.ok_or_else(|| de::Error::missing_field("model"))?;
        if model.to_lowercase() == C::NAME.to_lowercase() {
            Ok(ImageBase {
                data,
                model: PhantomData,
            })
        } else {
            Err(de::Error::invalid_value(
                de::Unexpected::Str(model),
                &C::NAME,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use crate::core::{Image, RGB};

    #[test]
    fn serialize_image_base() {
        const EXPECTED: &str = r#"{"data":{"v":1,"dim":[2,3,3],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"model":"RGB"}"#;
        let i = Image::<u8, RGB>::new(2, 3);
        let actual = serde_json::to_string(&i).expect("Serialized image");
        assert_eq!(actual, EXPECTED);
    }

    #[test]
    fn deserialize_image_base() {
        const EXPECTED: &str = r#"{"data":{"v":1,"dim":[2,3,3],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"model":"RGB"}"#;
        let actual: Image<u8, RGB> = serde_json::from_str(EXPECTED).expect("Deserialized image");
        assert_eq!(actual.model, PhantomData);
    }
}
