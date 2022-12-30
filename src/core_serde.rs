use crate::core::{ColourModel, ImageBase};
use ndarray::Data;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

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

#[cfg(test)]
mod tests {
    use crate::core::{Image, RGB};

    #[test]
    fn serialize_image_base() {
        const EXPECTED: &str = r#"{"data":{"v":1,"dim":[2,3,3],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"model":"RGB"}"#;
        let i = Image::<u8, RGB>::new(2, 3);
        let actual = serde_json::to_string(&i).expect("Serialized RGB image");
        assert_eq!(actual, EXPECTED);
    }
}
