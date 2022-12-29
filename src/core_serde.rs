use crate::core::{ColourModel, ImageBase};
use ndarray::Data;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

impl<A, T, C> Serialize for ImageBase<T, C>
where
    A: Serialize,
    T: Data<Elem = A> + Serialize,
    C: ColourModel,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Image", 2)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("model", &self.model)?;
        state.end()
    }
}
